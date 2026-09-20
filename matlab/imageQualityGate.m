function [qualityReport, enhancedImg] = imageQualityGate(img, options)
% IMAGEQUALITYGATE - Automatic Fundus Image Quality Assessment & Adaptive Enhancement
% RetinaSetu: Coaxial AI Screening Pipeline for Diabetic Retinopathy
%
% Syntax:
%   [qualityReport, enhancedImg] = imageQualityGate(img)
%   [qualityReport, enhancedImg] = imageQualityGate(img, 'FocusThreshold', 18.5, ...)
%
% Inputs:
%   img     - RGB retinal fundus image (uint8 or double)
%   options - Optional name-value pairs for threshold customization
%
% Outputs:
%   qualityReport - Struct containing:
%       .verdict        - 'ACCEPT', 'ENHANCED', or 'REJECT'
%       .qualityScore   - Composite quality score [0 - 100]
%       .focusScore     - Modified Laplacian variance metric
%       .illumScore     - Illumination entropy and uniformity metric
%       .glareRatio     - Ratio of saturated coaxial flash pixels
%       .fovCompleteness- Percentage of retinal circular mask visible
%       .feedback       - Actionable recapture feedback string
%   enhancedImg   - Enhanced RGB fundus image (if borderline/accepted)
%
% Toolboxes Used: Image Processing Toolbox, Medical Imaging Toolbox

arguments
    img
    options.FocusThreshold (1,1) double = 15.0      % Minimum acceptable Laplacian variance
    options.GlareMaxRatio (1,1) double = 0.045     % Maximum acceptable saturated glare fraction
    options.MinFovArea (1,1) double = 0.40         % Minimum field-of-view retinal area fraction
    options.ClaheClipLimit (1,1) double = 0.02     % CLAHE enhancement clip limit
end

if ischar(img) || isstring(img)
    img = imread(img);
end

if size(img, 3) == 1
    rgbImg = repmat(img, [1 1 3]);
else
    rgbImg = img;
end

if ~isa(rgbImg, 'uint8')
    rgbImg = im2uint8(rgbImg);
end

[rows, cols, ~] = size(rgbImg);
greenChannel = rgbImg(:,:,2);
grayImg = rgb2gray(rgbImg);

%% 1. Retinal Field of View (FOV) Mask Estimation
% Detect the circular aperture characteristic of fundus / 20D lens optics
redChannel = rgbImg(:,:,1);
fovMask = (redChannel > 15) | (greenChannel > 15);
fovMask = imclose(fovMask, strel('disk', 15));
fovMask = imfill(fovMask, 'holes');
fovArea = sum(fovMask(:));
totalPixels = rows * cols;
fovCompleteness = (fovArea / totalPixels);

%% 2. Focus & Sharpness Assessment (Modified Laplacian Variance)
% High variance indicates crisp microvasculature edges; low variance indicates blur
lapKernel = [0 1 0; 1 -4 1; 0 1 0];
lapFiltered = imfilter(double(greenChannel), lapKernel, 'replicate');
% Evaluate only inside FOV
lapFovValues = lapFiltered(fovMask);
if isempty(lapFovValues)
    focusScore = 0;
else
    focusScore = var(lapFovValues);
end

%% 3. Illumination Uniformity & Glare Detection
% Coaxial flash setups frequently produce central or off-axis reflections
glarePixels = (rgbImg(:,:,1) > 240) & (rgbImg(:,:,2) > 240) & (rgbImg(:,:,3) > 220) & fovMask;
glareRatio = sum(glarePixels(:)) / max(1, fovArea);

% Illumination entropy inside FOV
fovGreen = greenChannel(fovMask);
if isempty(fovGreen)
    illumScore = 0;
else
    illumScore = entropy(fovGreen);
end

%% 4. Composite Quality Index (0 - 100)
% Normalized weighted score
normFocus = min(1.0, focusScore / 40.0);
normIllum = min(1.0, illumScore / 7.5);
normGlare = max(0.0, 1.0 - (glareRatio / options.GlareMaxRatio));
normFov   = min(1.0, fovCompleteness / 0.70);

qualityScore = (0.40 * normFocus + 0.25 * normIllum + 0.20 * normGlare + 0.15 * normFov) * 100;

%% 5. Gating Decision & Recapture Feedback
feedbackMsgs = {};

if fovCompleteness < options.MinFovArea
    verdict = 'REJECT';
    feedbackMsgs{end+1} = 'FIELD OF VIEW: Retina off-centre or blocked. Align 20D lens coaxial with phone axis.';
elseif glareRatio > options.GlareMaxRatio
    verdict = 'REJECT';
    feedbackMsgs{end+1} = sprintf('FLASH GLARE: Excessive coaxial reflection (%.1f%%). Tilt camera 5-10 deg or dim flash.', glareRatio*100);
elseif focusScore < options.FocusThreshold * 0.65
    verdict = 'REJECT';
    feedbackMsgs{end+1} = sprintf('SEVERE BLUR: Focus metric (%.1f) below diagnostic limit. Stabilize mount and autofocus on optic disc.', focusScore);
elseif focusScore < options.FocusThreshold || qualityScore < 60
    verdict = 'ENHANCED';
    feedbackMsgs{end+1} = 'BORDERLINE QUALITY: Applying adaptive CLAHE and illumination normalization.';
else
    verdict = 'ACCEPT';
    feedbackMsgs{end+1} = 'OPTIMAL QUALITY: Passed automated quality gate for diagnostic grading.';
end

%% 6. Adaptive Enhancement Pipeline (CLAHE + Illumination Normalization)
if strcmp(verdict, 'ENHANCED') || strcmp(verdict, 'ACCEPT')
    % Convert to L*a*b* color space to enhance luminance without distorting chroma
    labImg = rgb2lab(rgbImg);
    L = labImg(:,:,1) / 100.0;
    
    % Contrast-Limited Adaptive Histogram Equalization
    L_enhanced = adapthisteq(L, 'ClipLimit', options.ClaheClipLimit, ...
                                'Distribution', 'rayleigh', ...
                                'NumTiles', [8 8]);
    
    % Illumination field flattening via large morphological opening
    bgEstimate = imopen(L_enhanced, strel('disk', 30));
    L_norm = L_enhanced - bgEstimate + mean(L_enhanced(fovMask));
    L_norm = max(0, min(1, L_norm));
    
    % Mild guided denoising
    if exist('imguidedfilter', 'file')
        L_final = imguidedfilter(L_norm);
    else
        L_final = medfilt2(L_norm, [3 3]);
    end
    
    labImg(:,:,1) = L_final * 100.0;
    enhancedImg = lab2rgb(labImg);
    enhancedImg = im2uint8(enhancedImg);
else
    enhancedImg = rgbImg; % Unusable as-is
end

%% Package Output Report
qualityReport.verdict = verdict;
qualityReport.qualityScore = round(qualityScore, 1);
qualityReport.focusScore = round(focusScore, 2);
qualityReport.illumScore = round(illumScore, 2);
qualityReport.glareRatio = round(glareRatio * 100, 2); % in percent
qualityReport.fovCompleteness = round(fovCompleteness * 100, 1);
qualityReport.feedback = strjoin(feedbackMsgs, ' ');

end
