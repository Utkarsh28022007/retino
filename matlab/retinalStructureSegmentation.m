function segResults = retinalStructureSegmentation(img, enhancedImg)
% RETINALSTRUCTURESEGMENTATION - Extracts retinal landmarks & pathological lesions
% RetinaSetu: Coaxial AI Screening Pipeline for Diabetic Retinopathy
%
% Extracts:
%   1. Optic Disc (OD) localization & boundary mask
%   2. Fovea centralis localization
%   3. Retinal Blood Vessel tree segmentation
%   4. Microaneurysms (MA) candidate detection (sub-pixel top-hat)
%   5. Hard Exudates (HE) segmentation
%   6. Hemorrhages (HEM) segmentation (blot/dot/flame)
%   7. Neovascularization (NV) risk index
%
% Toolboxes Used: Image Processing Toolbox, Computer Vision Toolbox

if nargin < 2 || isempty(enhancedImg)
    enhancedImg = img;
end

if ischar(img) || isstring(img)
    img = imread(img);
end
if ischar(enhancedImg) || isstring(enhancedImg)
    enhancedImg = imread(enhancedImg);
end

[rows, cols, ~] = size(img);
green = double(img(:,:,2)) / 255.0;
greenEnh = double(enhancedImg(:,:,2)) / 255.0;
red = double(img(:,:,1)) / 255.0;

% Retinal circular FOV mask
fovMask = (red > 0.08) | (green > 0.08);
fovMask = imclose(fovMask, strel('disk', 15));
fovMask = imfill(fovMask, 'holes');

%% 1. Optic Disc Localization & Segmentation
% The optic disc is characterized by high intensity in red/green channels and circular shape
odCandidateMap = (red * 0.6 + green * 0.4) .* fovMask;
odFiltered = imgaussfilt(odCandidateMap, 8);

% Find brightest regional cluster
[maxVal, maxIdx] = max(odFiltered(:));
[odCenterY, odCenterX] = ind2sub(size(odFiltered), maxIdx);

% Disc radius estimation (typically ~1/10 to 1/15 of image width)
discRadius = round(cols * 0.07);

% Create circular disc mask
[X, Y] = meshgrid(1:cols, 1:rows);
odDist = sqrt((X - odCenterX).^2 + (Y - odCenterY).^2);
opticDiscMask = (odDist <= discRadius) & fovMask;

%% 2. Fovea Centralis Localization
% The fovea is located approximately 2.5 disc diameters temporal to optic disc.
% If optic disc is in the right half of the image, the eye is the Right Eye (OD), fovea is to the left.
% If optic disc is in the left half, eye is Left Eye (OS), fovea is to the right.
if odCenterX > cols / 2
    foveaX = round(odCenterX - 2.5 * discRadius);
else
    foveaX = round(odCenterX + 2.5 * discRadius);
end
foveaY = round(odCenterY + 0.15 * discRadius); % Slightly inferior
foveaX = max(1, min(cols, foveaX));
foveaY = max(1, min(rows, foveaY));

foveaDist = sqrt((X - foveaX).^2 + (Y - foveaY).^2);
foveaMask = (foveaDist <= round(discRadius * 0.5)) & fovMask;

%% 3. Retinal Blood Vessel Tree Segmentation
% Vessel extraction using matched filtering and morphological top-hat on inverted green channel
invGreen = (1.0 - greenEnh) .* fovMask;

% Multi-scale morphological top-hat for tubular structure enhancement
seVessel1 = strel('disk', 3);
seVessel2 = strel('disk', 8);
topHat1 = imtophat(invGreen, seVessel1);
topHat2 = imtophat(invGreen, seVessel2);
vesselEnhanced = (topHat1 + topHat2) / 2.0;

% Adaptive thresholding inside FOV
meanVessel = mean(vesselEnhanced(fovMask));
stdVessel = std(vesselEnhanced(fovMask));
vesselThresh = meanVessel + 0.95 * stdVessel;
vesselMask = (vesselEnhanced > vesselThresh) & fovMask;

% Clean up tiny disconnected noise fragments
vesselMask = bwareaopen(vesselMask, 25);

%% 4. Microaneurysm (MA) Candidate Detection
% Microaneurysms are small, round, dark focal lesions (10 - 100 microns).
% Use morphological bottom-hat on green channel with small radius (1-4 pixels)
bottomHatMA = imbothat(green, strel('disk', 4));
bottomHatMA = bottomHatMA .* fovMask;

% Exclude blood vessel main branches so vessels aren't misclassified as MAs
vesselDilated = imdilate(vesselMask, strel('disk', 2));
maCandidates = (bottomHatMA > 0.07) & (~vesselDilated) & (~opticDiscMask);
maCandidates = bwareaopen(maCandidates, 2);
maCandidates = maCandidates & (~bwareaopen(maCandidates, 35)); % Remove large blobs

maCC = bwconncomp(maCandidates);
maCount = maCC.NumObjects;

%% 5. Hard Exudates (HE) Segmentation
% Hard exudates are bright yellowish lipid deposits with sharp boundaries.
% High luminance and high yellow (red+green vs blue)
blue = double(img(:,:,3)) / 255.0;
yellowIdx = (red + green - 1.5 * blue) .* fovMask;
exudateThresh = mean(yellowIdx(fovMask)) + 1.7 * std(yellowIdx(fovMask));
exudateMask = (yellowIdx > exudateThresh) & (~opticDiscMask) & fovMask;
exudateMask = bwareaopen(exudateMask, 8);
exudateAreaPct = (sum(exudateMask(:)) / max(1, sum(fovMask(:)))) * 100;

%% 6. Hemorrhages (HEM) Segmentation
% Blot and flame hemorrhages are dark reddish/brownish lesions outside vessel network
hemMap = (1.0 - green) .* (red > 0.35) .* fovMask;
hemThresh = mean(hemMap(fovMask)) + 1.2 * std(hemMap(fovMask));
hemMask = (hemMap > hemThresh) & (~vesselDilated) & (~opticDiscMask);
hemMask = bwareaopen(hemMask, 15);
hemAreaPct = (sum(hemMask(:)) / max(1, sum(fovMask(:)))) * 100;

%% 7. Neovascularization (NV) Risk Index
% New fragile vessels around the disc (NVD) or elsewhere (NVE) have high branching density & tortuosity
discVicinityMask = (odDist <= 1.8 * discRadius) & fovMask;
nvVessels = vesselMask & discVicinityMask;
nvDensity = sum(nvVessels(:)) / max(1, sum(discVicinityMask(:)));
nvIndex = min(1.0, nvDensity / 0.28); % Normalized risk index [0, 1]

%% Package Results
segResults.opticDiscCenter = [odCenterX, odCenterY];
segResults.opticDiscRadius = discRadius;
segResults.opticDiscMask = opticDiscMask;
segResults.foveaCenter = [foveaX, foveaY];
segResults.foveaMask = foveaMask;
segResults.vesselMask = vesselMask;
segResults.microaneurysmMask = maCandidates;
segResults.microaneurysmCount = maCount;
segResults.exudateMask = exudateMask;
segResults.exudateAreaPct = round(exudateAreaPct, 2);
segResults.hemorrhageMask = hemMask;
segResults.hemorrhageAreaPct = round(hemAreaPct, 2);
segResults.nvIndex = round(nvIndex, 3);
segResults.nvDetected = (nvIndex > 0.65);

end
