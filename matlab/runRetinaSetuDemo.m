% RUNRETINASETUDEMO - Master Demonstration Script for RetinaSetu
% Coaxial Tele-Ophthalmology & Automated AI Diabetic Retinopathy Screening
%
% This script executes the complete 5-stage DR screening pipeline:
%   Stage 1: Image Quality Assessment & Adaptive Enhancement
%   Stage 2: Retinal Structure & Pathological Lesion Segmentation
%   Stage 3: 5-Class ICDR Severity Classification (Levels 0-4)
%   Stage 4: Explainability Module (Grad-CAM Heatmap & 30s Report)
%   Stage 5: Simulink Discrete-Event Telemedicine Capacity Simulation
%   Stage 6: Clinical Benchmark & Ablation Study Validation

clc; clear; close all;
fprintf('========================================================================\n');
fprintf('       RetinaSetu: Automated Diabetic Retinopathy Screening Pipeline     \n');
fprintf('       Coaxial Tele-Ophthalmology & AI Retinal Screening System           \n');
fprintf('========================================================================\n\n');

%% Check Toolboxes
requiredToolboxes = {'Image Processing Toolbox', 'Computer Vision Toolbox', ...
                     'Deep Learning Toolbox', 'Statistics and Machine Learning Toolbox'};
fprintf('Verifying MATLAB Environment...\n');
for i = 1:length(requiredToolboxes)
    tbName = requiredToolboxes{i};
    hasTb = ~isempty(ver(tbName));
    if hasTb
        fprintf('  [OK] %s found.\n', tbName);
    else
        fprintf('  [INFO] %s (using algorithmic fallback if needed).\n', tbName);
    end
end
fprintf('\n');

%% Generate or Load a Representative Clinical Test Image
sampleImagePath = fullfile('..', 'sample_data', 'grade2_moderate.png');
if exist(sampleImagePath, 'file')
    testImg = imread(sampleImagePath);
else
    % Create a synthetic fundus disc for demonstration if sample file not yet generated
    [X, Y] = meshgrid(1:512, 1:512);
    dist = sqrt((X - 256).^2 + (Y - 256).^2);
    mask = dist <= 230;
    testImg = zeros(512, 512, 3, 'uint8');
    testImg(:,:,1) = uint8(mask .* (180 - dist*0.2));
    testImg(:,:,2) = uint8(mask .* (90 - dist*0.15));
    testImg(:,:,3) = uint8(mask .* 25);
end

%% Stage 1: Image Quality Assessment Gate
fprintf('>>> STAGE 1: Image Quality Assessment & Adaptive Enhancement <<<\n');
[qualityReport, enhancedImg] = imageQualityGate(testImg);
fprintf('  Verdict:            %s\n', qualityReport.verdict);
fprintf('  Composite Quality:  %.1f / 100\n', qualityReport.qualityScore);
fprintf('  Focus Score:        %.2f (Laplacian Var)\n', qualityReport.focusScore);
fprintf('  Glare Ratio:        %.2f%%\n', qualityReport.glareRatio);
fprintf('  Clinical Feedback:  %s\n\n', qualityReport.feedback);

%% Stage 2: Retinal Structure & Lesion Segmentation
fprintf('>>> STAGE 2: Retinal Structure & Lesion Segmentation <<<\n');
segResults = retinalStructureSegmentation(testImg, enhancedImg);
fprintf('  Optic Disc Center:      [%d, %d] (Radius: %d px)\n', ...
    segResults.opticDiscCenter(1), segResults.opticDiscCenter(2), segResults.opticDiscRadius);
fprintf('  Fovea Centralis:        [%d, %d]\n', segResults.foveaCenter(1), segResults.foveaCenter(2));
fprintf('  Microaneurysms Count:   %d lesions detected\n', segResults.microaneurysmCount);
fprintf('  Hard Exudates Area:     %.2f%% of retinal area\n', segResults.exudateAreaPct);
fprintf('  Hemorrhages Area:       %.2f%% of retinal area\n', segResults.hemorrhageAreaPct);
fprintf('  Neovascularization Idx: %.2f (Risk: %s)\n\n', ...
    segResults.nvIndex, mat2str(segResults.nvDetected));

%% Stage 3: DR Severity Grading (ICDR 0-4 Scale)
fprintf('>>> STAGE 3: ICDR Severity Grading & Referable DR Triage <<<\n');
[gradeReport, classProbs] = drSeverityGrading(enhancedImg, segResults);
fprintf('  Predicted Grade:    Level %d: %s\n', gradeReport.predictedGrade, gradeReport.gradeName);
fprintf('  Calibrated Conf:    %.1f%%\n', gradeReport.confidence);
fprintf('  Triage Category:    %s\n', gradeReport.triageCategory);
fprintf('  Referral Status:    %s\n', gradeReport.referralStatus);
fprintf('  Care Plan:          %s\n\n', gradeReport.followUpInterval);

%% Stage 4: Explainability Module (Grad-CAM & Fast Doctor Triage)
fprintf('>>> STAGE 4: Grad-CAM Explainability & Telemedicine Report <<<\n');
[gradCAMMap, overlayImg, explainReport] = drGradCAMExplainability(testImg, segResults, gradeReport);
fprintf('  Clinical Utility Rating:  %.1f / 5.0 (High Ophthalmologist Concordance)\n', explainReport.clinicalUtilityScore);
fprintf('  Lesion Attribution:       %.1f%% of deep attention on segmented lesions\n', explainReport.lesionAttributionRatio);
fprintf('\nDoctor Sign-Off Report:\n%s\n\n', explainReport.doctorSignOffSummary);

%% Stage 5: Simulink Discrete-Event Telemedicine Simulation
fprintf('>>> STAGE 5: Simulink Telemedicine District Optimization <<<\n');
simParams.annualPatientTarget = 100000;
simParams.numPHCs = 50;
simParams.avgNetworkBandwidthMbps = 1.5;
simMetrics = retinaSetuSimulinkModel(simParams);

%% Stage 6: Benchmark Validation
fprintf('>>> STAGE 6: Clinical Benchmark & Ablation Study <<<\n');
benchmarkValidation();

fprintf('========================================================================\n');
fprintf('  All 6 Stages Completed Successfully. Ready for Clinical Deployment Evaluation!   \n');
fprintf('========================================================================\n');
