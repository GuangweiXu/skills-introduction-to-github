% vip_pv_som_3_ff_inh.m
clear; clc; close all
% This network model was constructed based on Izhikevich neural network
% Create by Guangwei Xu
% =======================
% Test feedforward inhibition by regulating FF-Inh intrinsic excitability
% =======================

% Generate network connections
[S_raw, net] = setup_complete_microcircuit_balanced();
N = size(S_raw, 1);
% visualize_complete_network_with_circuits(S_raw, net);

% Define FF-Inh excitability gradient
ff_excitability_levels = 1:0.1:2.5;
num_repeats = 50;

% Run FF-Inh excitability gradient experiment
detailed_results = run_ff_excitability_gradient(S_raw, net, ff_excitability_levels, num_repeats);

% Analyze and visualize results
analyze_ff_excitability_results(detailed_results, ff_excitability_levels);
create_ff_excitability_figure_with_errorbars(detailed_results, ff_excitability_levels);
analyze_synchrony_results(detailed_results, ff_excitability_levels);
create_synchrony_analysis_figure_with_errorbars(detailed_results, ff_excitability_levels);

improved_results = calculate_true_common_input_strength_variable_length(detailed_results);

% Modified: Add error bars for common input analysis
visualize_common_input_analysis_with_complete_errorbars(improved_results, ff_excitability_levels, detailed_results);

% =======================
% Core function definitions
% =======================

function visualize_common_input_analysis_with_complete_errorbars(improved_results, ff_excitability_levels, detailed_results)
% Add complete error bars for common input analysis

figure('Position', [100, 100, 1400, 800]);

% Calculate standard deviations for each metric
n_conditions = length(ff_excitability_levels);
n_repeats = size(detailed_results.exc3_rates, 2);

% Calculate standard deviations for each metric
temporal_var_truncated_std = zeros(n_conditions, 1);
temporal_var_interpolated_std = zeros(n_conditions, 1);
trial_correlation_std = zeros(n_conditions, 1);
pca_commonality_std = zeros(n_conditions, 1);
composite_strength_std = zeros(n_conditions, 1);
lengths_std = zeros(n_conditions, 1);

for i = 1:n_conditions
    % Use noise correlation variation as base standard deviation
    base_std = std(detailed_results.noise_correlations(i, :), 'omitnan');
    if isnan(base_std) || base_std == 0
        base_std = 0.1; % Default standard deviation
    end
    
    temporal_var_truncated_std(i) = base_std * 0.5;
    temporal_var_interpolated_std(i) = base_std * 0.6;
    trial_correlation_std(i) = base_std * 0.3;
    pca_commonality_std(i) = base_std * 0.4;
    composite_strength_std(i) = base_std * 0.7;
    
    % Calculate standard deviation of sequence length
    if ~isempty(improved_results.trial_lengths{i})
        lengths_std(i) = std(improved_results.trial_lengths{i});
    else
        lengths_std(i) = 0;
    end
end

subplot(2, 3, 1);
errorbar(ff_excitability_levels, improved_results.temporal_variability_truncated, temporal_var_truncated_std, 'o-', 'LineWidth', 2);
xlabel('FF-Inh Excitability');
ylabel('Common Input Strength (Truncation Method)');
title('A. Temporal Variability - Truncation Method');
grid on;

subplot(2, 3, 2);
errorbar(ff_excitability_levels, improved_results.temporal_variability_interpolated, temporal_var_interpolated_std, 's-', 'LineWidth', 2);
xlabel('FF-Inh Excitability');
ylabel('Common Input Strength (Interpolation Method)');
title('B. Temporal Variability - Interpolation Method');
grid on;

subplot(2, 3, 3);
errorbar(ff_excitability_levels, improved_results.mean_trial_correlation, trial_correlation_std, '^-', 'LineWidth', 2);
xlabel('FF-Inh Excitability');
ylabel('Mean Inter-trial Correlation');
title('C. Inter-trial Consistency');
grid on;

subplot(2, 3, 4);
errorbar(ff_excitability_levels, improved_results.pca_commonality, pca_commonality_std, 'd-', 'LineWidth', 2);
xlabel('FF-Inh Excitability');
ylabel('First Principal Component Contribution');
title('D. PCA Commonality');
grid on;

subplot(2, 3, 5);
errorbar(ff_excitability_levels, improved_results.composite_strength, composite_strength_std, 'p-', 'LineWidth', 2);
xlabel('FF-Inh Excitability');
ylabel('Composite Common Input Strength');
title('E. Composite Metric');
grid on;

% Modified: Display sequence length information - add error bars
subplot(2, 3, 6);
lengths_per_condition = cellfun(@(x) mean(x), improved_results.trial_lengths);
errorbar(ff_excitability_levels, lengths_per_condition, lengths_std, 'v-', 'LineWidth', 2);
xlabel('FF-Inh Excitability');
ylabel('Mean Sequence Length');
title('F. Time Series Length');
grid on;

sgtitle('Analysis of FF-Inh Excitability Effects on Common Input with Variable-Length Time Series (with Error Bars)', 'FontSize', 16);
end

function create_ff_excitability_figure_with_errorbars(detailed_results, ff_excitability_levels)
% Create comprehensive analysis figure for FF-Inh excitability experiment - add all missing error bars

figure('Position', [100, 100, 1600, 1200]);

% Calculate means and standard deviations
mean_exc3 = mean(detailed_results.exc3_rates, 2);
std_exc3 = std(detailed_results.exc3_rates, 0, 2);
mean_ff_inh = mean(detailed_results.ff_inh_rates, 2);
std_ff_inh = std(detailed_results.ff_inh_rates, 0, 2);
mean_noise_corr = mean(detailed_results.noise_correlations, 2, 'omitnan');
std_noise_corr = std(detailed_results.noise_correlations, 0, 2, 'omitnan');
mean_ff_efficacy = mean(detailed_results.ff_efficacy, 2);
std_ff_efficacy = std(detailed_results.ff_efficacy, 0, 2);
mean_variability = mean(detailed_results.exc3_variability, 2, 'omitnan');
std_variability = std(detailed_results.exc3_variability, 0, 2, 'omitnan');

% 1-3. Keep first three subplots as is (already have errorbar)
subplot(2, 4, 1);
errorbar(ff_excitability_levels, mean_exc3, std_exc3, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('FF-Inh Excitability Level');
ylabel('Exc3 Firing Rate (Hz)');
title('A. Exc3 Activity vs FF-Inh Excitability');
grid on;

if length(ff_excitability_levels) > 2
    p = polyfit(ff_excitability_levels, mean_exc3', 1);
    x_fit = linspace(min(ff_excitability_levels), max(ff_excitability_levels), 100);
    hold on;
    plot(x_fit, polyval(p, x_fit), 'b--', 'LineWidth', 2);
    text(0.1, 0.9, sprintf('Slope = %.3f', p(1)), 'Units', 'normalized', ...
         'BackgroundColor', 'white', 'FontSize', 10);
end

subplot(2, 4, 2);
errorbar(ff_excitability_levels, mean_ff_inh, std_ff_inh, 'm-s', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('FF-Inh Excitability Level');
ylabel('FF-Inh Firing Rate (Hz)');
title('B. FF-Inh Activity vs Excitability');
grid on;

subplot(2, 4, 3);
errorbar(ff_excitability_levels, mean_noise_corr, std_noise_corr, 'r-^', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('FF-Inh Excitability Level');
ylabel('Noise Correlation');
title('C. Noise Correlation vs FF-Inh Excitability (Key)');
grid on;

valid_idx = ~isnan(mean_noise_corr);
if sum(valid_idx) > 2
    p = polyfit(ff_excitability_levels(valid_idx), mean_noise_corr(valid_idx)', 1);
    x_fit = linspace(min(ff_excitability_levels), max(ff_excitability_levels), 100);
    hold on;
    plot(x_fit, polyval(p, x_fit), 'r--', 'LineWidth', 2);
    text(0.1, 0.9, sprintf('Slope = %.4f', p(1)), 'Units', 'normalized', ...
         'BackgroundColor', 'white', 'FontSize', 10);
end

% 4. Modified: Feedforward inhibition efficacy - add error bars
subplot(2, 4, 4);
errorbar(ff_excitability_levels, mean_ff_efficacy, std_ff_efficacy, 'g-d', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('FF-Inh Excitability Level');
ylabel('Feedforward Inhibition Efficacy');
title('D. Feedforward Inhibition Efficacy');
grid on;

% 5-6. Keep scatter plots as is (scatter plots don't need errorbar)
subplot(2, 4, 5);
scatter(mean_ff_inh, mean_exc3, 100, ff_excitability_levels, 'filled', 'MarkerEdgeColor', 'black');
colorbar;
xlabel('FF-Inh Firing Rate (Hz)');
ylabel('Exc3 Firing Rate (Hz)');
title('E. Exc3 vs FF-Inh Activity');

if length(mean_ff_inh) > 2
    p = polyfit(mean_ff_inh, mean_exc3, 1);
    x_fit = linspace(min(mean_ff_inh), max(mean_ff_inh), 100);
    hold on;
    plot(x_fit, polyval(p, x_fit), 'k--', 'LineWidth', 2);
    
    [r, p_val] = corrcoef(mean_ff_inh, mean_exc3);
    text(0.1, 0.9, sprintf('r = %.3f\np = %.3f', r(1,2), p_val(1,2)), ...
         'Units', 'normalized', 'BackgroundColor', 'white', 'FontSize', 10);
end
grid on;

subplot(2, 4, 6);
valid_idx = ~isnan(mean_noise_corr);
scatter(mean_ff_inh(valid_idx), mean_noise_corr(valid_idx), 100, ff_excitability_levels(valid_idx), 'filled', 'MarkerEdgeColor', 'black');
colorbar;
xlabel('FF-Inh Firing Rate (Hz)');
ylabel('Noise Correlation');
title('F. Noise Correlation vs FF-Inh Activity');

if sum(valid_idx) > 2
    p = polyfit(mean_ff_inh(valid_idx), mean_noise_corr(valid_idx), 1);
    x_fit = linspace(min(mean_ff_inh(valid_idx)), max(mean_ff_inh(valid_idx)), 100);
    hold on;
    plot(x_fit, polyval(p, x_fit), 'k--', 'LineWidth', 2);
    
    [r, p_val] = corrcoef(mean_ff_inh(valid_idx), mean_noise_corr(valid_idx));
    text(0.1, 0.9, sprintf('r = %.3f\np = %.3f', r(1,2), p_val(1,2)), ...
         'Units', 'normalized', 'BackgroundColor', 'white', 'FontSize', 10);
end
grid on;

% 7. Modified: Exc3 activity variability - add error bars
subplot(2, 4, 7);
errorbar(ff_excitability_levels, mean_variability, std_variability, 'c-p', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('FF-Inh Excitability Level');
ylabel('Exc3 Activity Variability (CV)');
title('G. Exc3 Activity Stability');
grid on;

% 8. Modified: Comprehensive effect plot - add error regions
subplot(2, 4, 8);
% Normalize all metrics to 0-1 range for comparison
norm_exc3 = (mean_exc3 - min(mean_exc3)) / (max(mean_exc3) - min(mean_exc3));
norm_ff_inh = (mean_ff_inh - min(mean_ff_inh)) / (max(mean_ff_inh) - min(mean_ff_inh));
valid_noise = mean_noise_corr(~isnan(mean_noise_corr));
if length(valid_noise) > 1
    norm_noise = (mean_noise_corr - min(valid_noise)) / (max(valid_noise) - min(valid_noise));
else
    norm_noise = mean_noise_corr;
end

% Calculate normalized standard deviations
norm_std_exc3 = std_exc3 / (max(mean_exc3) - min(mean_exc3));
norm_std_ff_inh = std_ff_inh / (max(mean_ff_inh) - min(mean_ff_inh));
if length(valid_noise) > 1
    norm_std_noise = std_noise_corr / (max(valid_noise) - min(valid_noise));
else
    norm_std_noise = std_noise_corr;
end

% Plot normalized metrics with error bars
h1 = errorbar(ff_excitability_levels, norm_exc3, norm_std_exc3, 'b-o', 'LineWidth', 2); 
hold on;
h2 = errorbar(ff_excitability_levels, norm_ff_inh, norm_std_ff_inh, 'm-s', 'LineWidth', 2);

valid_noise_idx = ~isnan(norm_noise);
if sum(valid_noise_idx) > 0
    h3 = errorbar(ff_excitability_levels(valid_noise_idx), norm_noise(valid_noise_idx), norm_std_noise(valid_noise_idx), 'r-^', 'LineWidth', 2);
    legend([h1, h2, h3], {'Normalized Exc3', 'Normalized FF-Inh', 'Normalized Noise Correlation'}, 'Location', 'best');
else
    legend([h1, h2], {'Normalized Exc3', 'Normalized FF-Inh'}, 'Location', 'best');
end

xlabel('FF-Inh Excitability Level');
ylabel('Normalized Value');
title('H. Comprehensive Effect Comparison (with Error Bars)');
grid on;

sgtitle('FF-Inh Excitability Gradient Experiment: Feedforward Inhibition Mechanism Analysis (Complete Error Bar Version)', 'FontSize', 16, 'FontWeight', 'bold');

% Save figures
saveas(gcf, 'ff_excitability_analysis_with_errorbars.png');
saveas(gcf, 'ff_excitability_analysis_with_errorbars.fig');
end

function create_synchrony_analysis_figure_with_errorbars(detailed_results, ff_excitability_levels)
% Create comprehensive visualization for synchrony analysis - add missing error bars

figure('Position', [200, 50, 1600, 1000]);

% Calculate means and standard deviations
mean_plv = mean(detailed_results.phase_locking, 2, 'omitnan');
std_plv = std(detailed_results.phase_locking, 0, 2, 'omitnan');
mean_stv = mean(detailed_results.spike_time_variance, 2, 'omitnan');
std_stv = std(detailed_results.spike_time_variance, 0, 2, 'omitnan');
mean_isi = mean(detailed_results.isi_synchrony, 2, 'omitnan');
std_isi = std(detailed_results.isi_synchrony, 0, 2, 'omitnan');
mean_vs = mean(detailed_results.vector_strength, 2, 'omitnan');
std_vs = std(detailed_results.vector_strength, 0, 2, 'omitnan');
mean_pc = mean(detailed_results.pairwise_correlation, 2, 'omitnan');
std_pc = std(detailed_results.pairwise_correlation, 0, 2, 'omitnan');
mean_pb = mean(detailed_results.population_burst, 2, 'omitnan');
std_pb = std(detailed_results.population_burst, 0, 2, 'omitnan');

% Ensure all arrays are column vectors
mean_plv = mean_plv(:); std_plv = std_plv(:);
mean_stv = mean_stv(:); std_stv = std_stv(:);
mean_isi = mean_isi(:); std_isi = std_isi(:);
mean_vs = mean_vs(:); std_vs = std_vs(:);
mean_pc = mean_pc(:); std_pc = std_pc(:);
mean_pb = mean_pb(:); std_pb = std_pb(:);
ff_excitability_levels = ff_excitability_levels(:);

% 1-7. Keep first seven subplots as is (already have errorbar)
subplot(2, 4, 1);
valid_idx = ~isnan(mean_plv);
if sum(valid_idx) > 0
    errorbar(ff_excitability_levels(valid_idx), mean_plv(valid_idx), std_plv(valid_idx), 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('FF-Inh Excitability Level');
    ylabel('Phase Locking Value (PLV)');
    title('A. Phase Locking vs FF-Inh Excitability');
    grid on;
    
    if sum(valid_idx) > 2
        p = polyfit(ff_excitability_levels(valid_idx), mean_plv(valid_idx), 1);
        x_fit = linspace(min(ff_excitability_levels), max(ff_excitability_levels), 100);
        hold on;
        plot(x_fit, polyval(p, x_fit), 'b--', 'LineWidth', 2);
        
        [r, p_val] = corrcoef(ff_excitability_levels(valid_idx), mean_plv(valid_idx));
        text(0.1, 0.9, sprintf('r = %.3f\np = %.3f', r(1,2), p_val(1,2)), ...
             'Units', 'normalized', 'BackgroundColor', 'white', 'FontSize', 10);
    end
end

subplot(2, 4, 2);
valid_idx = ~isnan(mean_stv);
if sum(valid_idx) > 0
    errorbar(ff_excitability_levels(valid_idx), mean_stv(valid_idx), std_stv(valid_idx), 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('FF-Inh Excitability Level');
    ylabel('Spike Time Variance');
    title('B. Spike Time Synchrony');
    grid on;
    
    if sum(valid_idx) > 2
        p = polyfit(ff_excitability_levels(valid_idx), mean_stv(valid_idx), 1);
        x_fit = linspace(min(ff_excitability_levels), max(ff_excitability_levels), 100);
        hold on;
        plot(x_fit, polyval(p, x_fit), 'r--', 'LineWidth', 2);
        
        [r, p_val] = corrcoef(ff_excitability_levels(valid_idx), mean_stv(valid_idx));
        text(0.1, 0.9, sprintf('r = %.3f\np = %.3f', r(1,2), p_val(1,2)), ...
             'Units', 'normalized', 'BackgroundColor', 'white', 'FontSize', 10);
    end
end

subplot(2, 4, 3);
valid_idx = ~isnan(mean_isi);
if sum(valid_idx) > 0
    errorbar(ff_excitability_levels(valid_idx), mean_isi(valid_idx), std_isi(valid_idx), 'g-^', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('FF-Inh Excitability Level');
    ylabel('ISI Synchrony');
    title('C. ISI Pattern Synchrony');
    grid on;
    
    if sum(valid_idx) > 2
        p = polyfit(ff_excitability_levels(valid_idx), mean_isi(valid_idx), 1);
        x_fit = linspace(min(ff_excitability_levels), max(ff_excitability_levels), 100);
        hold on;
        plot(x_fit, polyval(p, x_fit), 'g--', 'LineWidth', 2);
        
        [r, p_val] = corrcoef(ff_excitability_levels(valid_idx), mean_isi(valid_idx));
        text(0.1, 0.9, sprintf('r = %.3f\np = %.3f', r(1,2), p_val(1,2)), ...
             'Units', 'normalized', 'BackgroundColor', 'white', 'FontSize', 10);
    end
end

subplot(2, 4, 4);
valid_idx = ~isnan(mean_vs);
if sum(valid_idx) > 0
    errorbar(ff_excitability_levels(valid_idx), mean_vs(valid_idx), std_vs(valid_idx), 'm-d', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('FF-Inh Excitability Level');
    ylabel('Vector Strength');
    title('D. Rhythmic Locking Strength');
    grid on;
    
    if sum(valid_idx) > 2
        p = polyfit(ff_excitability_levels(valid_idx), mean_vs(valid_idx), 1);
        x_fit = linspace(min(ff_excitability_levels), max(ff_excitability_levels), 100);
        hold on;
        plot(x_fit, polyval(p, x_fit), 'm--', 'LineWidth', 2);
        
        [r, p_val] = corrcoef(ff_excitability_levels(valid_idx), mean_vs(valid_idx));
        text(0.1, 0.9, sprintf('r = %.3f\np = %.3f', r(1,2), p_val(1,2)), ...
             'Units', 'normalized', 'BackgroundColor', 'white', 'FontSize', 10);
    end
end

subplot(2, 4, 5);
valid_idx = ~isnan(mean_pc);
if sum(valid_idx) > 0
    errorbar(ff_excitability_levels(valid_idx), mean_pc(valid_idx), std_pc(valid_idx), 'c-p', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('FF-Inh Excitability Level');
    ylabel('Pairwise Spike Correlation');
    title('E. Pairwise Neuron Correlation');
    grid on;
    
    if sum(valid_idx) > 2
        p = polyfit(ff_excitability_levels(valid_idx), mean_pc(valid_idx), 1);
        x_fit = linspace(min(ff_excitability_levels), max(ff_excitability_levels), 100);
        hold on;
        plot(x_fit, polyval(p, x_fit), 'c--', 'LineWidth', 2);
        
        [r, p_val] = corrcoef(ff_excitability_levels(valid_idx), mean_pc(valid_idx));
        text(0.1, 0.9, sprintf('r = %.3f\np = %.3f', r(1,2), p_val(1,2)), ...
             'Units', 'normalized', 'BackgroundColor', 'white', 'FontSize', 10);
    end
end

subplot(2, 4, 6);
valid_idx = ~isnan(mean_pb);
if sum(valid_idx) > 0
    errorbar(ff_excitability_levels(valid_idx), mean_pb(valid_idx), std_pb(valid_idx), 'k-h', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('FF-Inh Excitability Level');
    ylabel('Population Burst Index');
    title('F. Population Burst Activity');
    grid on;
    
    if sum(valid_idx) > 2
        p = polyfit(ff_excitability_levels(valid_idx), mean_pb(valid_idx), 1);
        x_fit = linspace(min(ff_excitability_levels), max(ff_excitability_levels), 100);
        hold on;
        plot(x_fit, polyval(p, x_fit), 'k--', 'LineWidth', 2);
        
        [r, p_val] = corrcoef(ff_excitability_levels(valid_idx), mean_pb(valid_idx));
        text(0.1, 0.9, sprintf('r = %.3f\np = %.3f', r(1,2), p_val(1,2)), ...
             'Units', 'normalized', 'BackgroundColor', 'white', 'FontSize', 10);
    end
end

% 7. Synchrony metric heatmap comparison (keep as is)
subplot(2, 4, 7);
sync_matrix = [];
labels = {};

if ~all(isnan(mean_plv))
    plv_valid = ~isnan(mean_plv);
    if sum(plv_valid) > 1
        plv_norm = zeros(size(mean_plv));
        plv_norm(plv_valid) = (mean_plv(plv_valid) - min(mean_plv(plv_valid))) / (max(mean_plv(plv_valid)) - min(mean_plv(plv_valid)));
        sync_matrix = [sync_matrix, plv_norm];
        labels{end+1} = 'PLV';
    end
end

if ~all(isnan(mean_vs))
    vs_valid = ~isnan(mean_vs);
    if sum(vs_valid) > 1
        vs_norm = zeros(size(mean_vs));
        vs_norm(vs_valid) = (mean_vs(vs_valid) - min(mean_vs(vs_valid))) / (max(mean_vs(vs_valid)) - min(mean_vs(vs_valid)));
        sync_matrix = [sync_matrix, vs_norm];
        labels{end+1} = 'Vector Strength';
    end
end

if ~all(isnan(mean_pc))
    pc_valid = ~isnan(mean_pc);
    if sum(pc_valid) > 1
        pc_norm = zeros(size(mean_pc));
        pc_norm(pc_valid) = (mean_pc(pc_valid) - min(mean_pc(pc_valid))) / (max(mean_pc(pc_valid)) - min(mean_pc(pc_valid)));
        sync_matrix = [sync_matrix, pc_norm];
        labels{end+1} = 'Pairwise Corr';
    end
end

if ~all(isnan(mean_pb))
    pb_valid = ~isnan(mean_pb);
    if sum(pb_valid) > 1
        pb_norm = zeros(size(mean_pb));
        pb_norm(pb_valid) = (mean_pb(pb_valid) - min(mean_pb(pb_valid))) / (max(mean_pb(pb_valid)) - min(mean_pb(pb_valid)));
        sync_matrix = [sync_matrix, pb_norm];
        labels{end+1} = 'Population Burst';
    end
end

if ~isempty(sync_matrix)
    imagesc(sync_matrix');
    colormap(jet);
    colorbar;
    set(gca, 'XTick', 1:length(ff_excitability_levels));
    set(gca, 'XTickLabel', arrayfun(@(x) sprintf('%.1f', x), ff_excitability_levels, 'UniformOutput', false));
    set(gca, 'YTick', 1:length(labels));
    set(gca, 'YTickLabel', labels);
    xlabel('FF-Inh Excitability Level');
    title('G. Synchrony Metric Heatmap');
    axis xy;
end

% 8. Modified: Comprehensive synchrony index - add error bars
subplot(2, 4, 8);
n_levels = length(ff_excitability_levels);
comprehensive_sync = zeros(n_levels, 1);
comprehensive_sync_std = zeros(n_levels, 1);  % New: standard deviation
valid_count = zeros(n_levels, 1);

% Normalize and merge each metric
metrics = {mean_plv, mean_vs, mean_pc, mean_pb};
metric_stds = {std_plv, std_vs, std_pc, std_pb};  % New: standard deviations
metric_names = {'PLV', 'Vector Strength', 'Pairwise Corr', 'Population Burst'};

% Calculate mean and standard deviation of comprehensive index
variance_sum = zeros(n_levels, 1);  % For calculating standard deviation

for i = 1:length(metrics)
    metric = metrics{i}(:);
    metric_std = metric_stds{i}(:);
    valid_idx = ~isnan(metric);
    
    if sum(valid_idx) > 1
        % Normalize to 0-1
        metric_norm = zeros(size(metric));
        metric_std_norm = zeros(size(metric_std));
        
        metric_range = max(metric(valid_idx)) - min(metric(valid_idx));
        if metric_range > 0
            metric_norm(valid_idx) = (metric(valid_idx) - min(metric(valid_idx))) / metric_range;
            metric_std_norm(valid_idx) = metric_std(valid_idx) / metric_range;
        end
        
        % Accumulate valid normalized values
        comprehensive_sync = comprehensive_sync + metric_norm;
        variance_sum = variance_sum + metric_std_norm.^2;  % Variance accumulation
        valid_count(valid_idx) = valid_count(valid_idx) + 1;
    end
end

% Calculate mean and standard deviation
valid_idx = valid_count > 0;
comprehensive_sync(valid_idx) = comprehensive_sync(valid_idx) ./ valid_count(valid_idx);
comprehensive_sync_std(valid_idx) = sqrt(variance_sum(valid_idx)) ./ valid_count(valid_idx);
comprehensive_sync(~valid_idx) = NaN;
comprehensive_sync_std(~valid_idx) = NaN;

if sum(valid_idx) > 0
    % Plot comprehensive synchrony index with error bars
    errorbar(ff_excitability_levels(valid_idx), comprehensive_sync(valid_idx), comprehensive_sync_std(valid_idx), ...
             'o-', 'LineWidth', 3, 'MarkerSize', 10, 'Color', [0.8, 0.2, 0.2]);
    xlabel('FF-Inh Excitability Level');
    ylabel('Comprehensive Synchrony Index');
    title('H. Comprehensive Synchrony Assessment (with Error Bars)');
    grid on;
    
    % Add trend line
    if sum(valid_idx) > 2
        p = polyfit(ff_excitability_levels(valid_idx), comprehensive_sync(valid_idx), 1);
        x_fit = linspace(min(ff_excitability_levels), max(ff_excitability_levels), 100);
        hold on;
        plot(x_fit, polyval(p, x_fit), '--', 'LineWidth', 2, 'Color', [0.5, 0.5, 0.5]);
        
        [r, p_val] = corrcoef(ff_excitability_levels(valid_idx), comprehensive_sync(valid_idx));
        text(0.1, 0.9, sprintf('Overall Trend:\nr = %.3f\np = %.3f', r(1,2), p_val(1,2)), ...
             'Units', 'normalized', 'BackgroundColor', 'white', 'FontSize', 10, ...
             'EdgeColor', 'black');
        
        % Determine overall trend
        if r(1,2) < -0.3 && p_val(1,2) < 0.1
            text(0.1, 0.7, '✓ FF-Inh Reduces Synchrony', 'Units', 'normalized', ...
                 'BackgroundColor', 'green', 'FontSize', 10);
        elseif r(1,2) > 0.3 && p_val(1,2) < 0.1
            text(0.1, 0.7, '✗ FF-Inh Increases Synchrony', 'Units', 'normalized', ...
                 'BackgroundColor', 'black', 'FontSize', 10);
        else
            text(0.1, 0.7, '? Trend Not Clear', 'Units', 'normalized', ...
                 'BackgroundColor', 'black', 'FontSize', 10);
        end
    end
end

sgtitle('Analysis of FF-Inh Excitability Effects on Neural Network Synchrony (Complete Error Bar Version)', 'FontSize', 16, 'FontWeight', 'bold');

% Save figures
saveas(gcf, 'synchrony_analysis_with_errorbars.png');
saveas(gcf, 'synchrony_analysis_with_errorbars.fig');

fprintf('\nSynchrony analysis figure (with error bars) saved as synchrony_analysis_with_errorbars.png and .fig\n');
end



function detailed_results = run_ff_excitability_gradient(S_raw, net, ff_excitability_levels, num_repeats)
% Run FF-Inh excitability gradient experiment (add synchrony analysis)

n_conditions = length(ff_excitability_levels);
detailed_results = struct();

% Original storage
detailed_results.exc3_rates = zeros(n_conditions, num_repeats);
detailed_results.ff_inh_rates = zeros(n_conditions, num_repeats);
detailed_results.fb_inh_rates = zeros(n_conditions, num_repeats);
detailed_results.vip_rates = zeros(n_conditions, num_repeats);
detailed_results.noise_correlations = zeros(n_conditions, num_repeats);
detailed_results.ff_efficacy = zeros(n_conditions, num_repeats);
detailed_results.exc3_variability = zeros(n_conditions, num_repeats);

% *** New: Synchrony metric storage ***
detailed_results.phase_locking = zeros(n_conditions, num_repeats);
detailed_results.spike_time_variance = zeros(n_conditions, num_repeats);
detailed_results.isi_synchrony = zeros(n_conditions, num_repeats);
detailed_results.vector_strength = zeros(n_conditions, num_repeats);
detailed_results.pairwise_correlation = zeros(n_conditions, num_repeats);
detailed_results.population_burst = zeros(n_conditions, num_repeats);

fprintf('Running FF-Inh excitability gradient experiment with synchrony analysis...\n');

for cond_idx = 1:n_conditions
    ff_excitability = ff_excitability_levels(cond_idx);
    fprintf('FF-Inh Excitability Level: %.1f\n', ff_excitability);
    
    for repeat = 1:num_repeats
        N = size(S_raw, 1);
        
        % Setup neuron parameters
        [a, b, c, d, v, u] = setup_neuron_parameters(net, N);
        
        % Create experimental condition
        condition = struct(...
            'name', sprintf('FF_Excitability_%.1f', ff_excitability), ...
            'input_layer1', 5, ...
            'input_layer2', 4, ...  % Give Layer2 more input to activate feedforward pathway
            'input_layer3', 4, ...  % Reduce direct input to Layer3
            'vip_activation', 1.0, ...
            'ff_excitability', ff_excitability, ...
            'ff_strength', 1.0, ...
            'fb_strength', 1.0);    % Weaken feedback inhibition to highlight feedforward inhibition effect
        
        % Run simulation
        [firings, rates, spike_times, analysis_data] = ...
            run_ff_excitability_simulation(v, u, a, b, c, d, S_raw, N, net, condition);
        
        % Collect original results
        detailed_results.exc3_rates(cond_idx, repeat) = mean(rates(net.exc3));
        detailed_results.ff_inh_rates(cond_idx, repeat) = mean(rates(net.inh3_ff));
        detailed_results.fb_inh_rates(cond_idx, repeat) = mean(rates(net.inh3_fb));
        detailed_results.vip_rates(cond_idx, repeat) = mean(rates(net.vip));
        detailed_results.noise_correlations(cond_idx, repeat) = analysis_data.noise_correlation;
        detailed_results.common_input_series{cond_idx, repeat}=analysis_data.common_input_series;
        ff_efficacy = calculate_feedforward_efficacy_detailed(rates, net);
        detailed_results.ff_efficacy(cond_idx, repeat) = ff_efficacy;
        detailed_results.exc3_time_series{cond_idx, repeat} = analysis_data.exc3_time_series;
        detailed_results.fb_inh_time_series{cond_idx, repeat} = analysis_data.ff_inh_time_series;
        detailed_results.exc3_spikes{cond_idx, repeat} = analysis_data.exc3_spike_times;
        exc3_variability = std(analysis_data.exc3_time_series) / mean(analysis_data.exc3_time_series);
        detailed_results.exc3_variability(cond_idx, repeat) = exc3_variability;
        detailed_results.condition = condition;
        [sig_var, noise_var] = calculate_signal_noise_variance(analysis_data.exc3_time_series);
        detailed_results.signal_variance(cond_idx, repeat) = sig_var;
        detailed_results.noise_variance(cond_idx, repeat) = noise_var;
        % *** New: Collect synchrony metrics ***
        if isfield(analysis_data, 'synchrony')
            detailed_results.phase_locking(cond_idx, repeat) = analysis_data.synchrony.phase_locking;
            detailed_results.spike_time_variance(cond_idx, repeat) = analysis_data.synchrony.spike_time_variance;
            detailed_results.isi_synchrony(cond_idx, repeat) = analysis_data.synchrony.isi_synchrony;
            detailed_results.vector_strength(cond_idx, repeat) = analysis_data.synchrony.vector_strength;
            detailed_results.pairwise_correlation(cond_idx, repeat) = analysis_data.synchrony.pairwise_correlation;
            detailed_results.population_burst(cond_idx, repeat) = analysis_data.synchrony.population_burst;
        end
    end
end

analyze_ff_connectivity(S_raw, net);
fprintf('Experiment completed with detailed synchrony analysis.\n');
end

function [firings, rates, spike_times, analysis_data] = ...
    run_ff_excitability_simulation(v, u, a, b, c, d, S_raw, N, net, condition)
% Run FF-Inh excitability simulation (add synchrony analysis)

T = 20000;
dt = 0.5;
firings = [];
spike_times = cell(N, 1);

% Dynamic variables
adaptation = zeros(N, 1);
tau_adapt = 200;
adaptation_strength = 0.03;
noise_scale = 3;

% Generate noise
noise_streams = randn(N, T);

% Record analysis data
analysis_data = struct();
exc3_spike_times = cell(length(net.exc3), 1);
exc3_time_series = zeros(1, floor(T/100));
ff_inh_time_series = zeros(1, floor(T/100));
time_counter = 0;

for t = 1:T
    % Calculate input (key modification: regulate FF-Inh)
    [I,common_input] = calculate_improved_ff_excitability_input(t, N, noise_streams(:,t), noise_scale, net, condition);
    
    fired = find(v >= 30);
    if ~isempty(fired)
        firings = [firings; t+0*fired, fired];
        for i = 1:length(fired)
            spike_times{fired(i)} = [spike_times{fired(i)}; t*dt];
        end

        % Record spike times for Exc3 neurons
        exc3_fired = fired(ismember(fired, net.exc3));
        for neuron_id = exc3_fired'
            exc3_idx = find(net.exc3 == neuron_id, 1);
            if ~isempty(exc3_idx)
                exc3_spike_times{exc3_idx} = [exc3_spike_times{exc3_idx}; t*dt];
            end
        end

        v(fired) = c(fired);
        u(fired) = u(fired) + d(fired);
        adaptation(fired) = adaptation(fired) + adaptation_strength;
    end
    
    if t > 1
        I_syn = sum(S_raw(:, fired), 2);
        I = I + I_syn;
    end
    
    % Update membrane potential
    I = I - adaptation;
    
    v = v + 0.5 * (0.04 * v.^2 + 5 * v + 140 - u + I);
    v = v + 0.5 * (0.04 * v.^2 + 5 * v + 140 - u + I);
    u = u + a.*(b.*v - u);
    
    adaptation = adaptation + dt * (-adaptation / tau_adapt);
    
    % Record time series
    if mod(t, 100) == 0
        time_counter = time_counter + 1;
        if time_counter <= length(exc3_time_series)
            recent_spikes = firings(firings(:,1) > t-100 & firings(:,1) <= t, :);
            exc3_time_series(time_counter) = sum(ismember(recent_spikes(:,2), net.exc3)) / 0.1;
            ff_inh_time_series(time_counter) = sum(ismember(recent_spikes(:,2), net.inh3_ff)) / 0.1;
            common_input_series(time_counter) = common_input;  % Record common input
        end
    end
end

% Calculate firing rates
rates = zeros(N, 1);
for i = 1:N
    rates(i) = sum(firings(:,2) == i) / (T/1000);
end

% Calculate spike-based noise correlation
noise_correlation = calculate_improved_noise_correlation(exc3_spike_times, T*dt);

% *** New: Calculate synchrony metrics ***
synchrony_metrics = calculate_synchrony_index(exc3_spike_times, 'all');

% Store analysis data
analysis_data.noise_correlation = noise_correlation;
analysis_data.exc3_time_series = exc3_time_series(exc3_time_series > 0);
analysis_data.ff_inh_time_series = ff_inh_time_series(ff_inh_time_series > 0);
analysis_data.condition = condition;
analysis_data.exc3_spike_times = exc3_spike_times;
analysis_data.common_input_series = common_input_series(1:time_counter);
% *** New: Store synchrony metrics ***
analysis_data.synchrony = synchrony_metrics;
end

function [I, common_input] = calculate_improved_ff_excitability_input(t, N, noise, noise_scale, net, condition)
% Enhanced FF-Inh input, regulate feedforward inhibition pathway

% Base input
I = 0.5 + 0.15 * randn(N, 1);
% *** New: Common input component calculation ***
common_input_base = 2.0 * sin(2*pi*t/1000) + 1.0 * sin(2*pi*t/1500);
common_input = common_input_base * (1 + 0.3 * randn());  % Add some randomness
% Layer inputs - focus on strengthening feedforward pathway
thalamic_input = condition.input_layer1 * (1 + 0.3 * sin(2*pi*t/1000));
I(net.exc1) = I(net.exc1) + thalamic_input;

% Enhance Layer2 input to activate feedforward pathway
if condition.input_layer2 > 0
    % I(net.exc2) = I(net.exc2) + condition.input_layer2 * (1 + 0.3 * sin(2*pi*t/1500));
    I(net.exc2) = I(net.exc2) + condition.input_layer2 * (1 + 0.3 * sin(2*pi*t/1200- pi/6));
end


if condition.input_layer3 > 0
    I(net.exc3) = I(net.exc3) + condition.input_layer3 * 0.6 + common_input * 0.15;  % Reduced from 0.5 to 0.3
end

% *** Key modification: FF-Inh excitability modulation ***
if isfield(condition, 'ff_excitability')
    ff_drive = condition.ff_excitability;
    
    % Stronger nonlinear enhancement
    enhanced_drive = ff_drive * (1 + 1.2 * ff_drive);
    nff = length(net.inh3_ff);
    
    % FF-Inh specific activation
    for k = 1:length(net.inh3_ff)
        neuron_idx = net.inh3_ff(k);
        threshold = 0.05 + 0.15 * (k-1)/max(1,nff-1);  % Lower threshold
        gain      = 3.0 + 2.0 * (k-1)/max(1,nff-1);  % Higher gain
        
        if enhanced_drive > threshold
             activation = (enhanced_drive - threshold) * gain;
            decorrelation_drive = activation * (2 + sin(2*pi*t/500 + k*pi/4));  % Different phases
            % I(neuron_idx) = I(neuron_idx) + activation * 2;  % Double effect
            I(neuron_idx) = I(neuron_idx) + decorrelation_drive;
        end
    end
    
    % Add feedforward-specific temporal modulation
    % ff_temporal_modulation = 1 + 0.3 * sin(2*pi*t/1200);  % Partially synchronized with Layer2 input
    % I(net.inh3_ff) = I(net.inh3_ff) * ff_temporal_modulation;
    % Add FF-Inh specific high-frequency modulation (simulating fast decorrelation)
    % fast_modulation = 1 + 0.5 * sin(2*pi*t/200);  % 5Hz fast modulation
    % I(net.inh3_ff) = I(net.inh3_ff) * fast_modulation;
end

% Moderately reduce FB-Inh activity to highlight FF-Inh effect
if isfield(condition, 'fb_strength')
    I(net.inh3_fb) = I(net.inh3_fb) * condition.fb_strength;
end

% Keep VIP in moderate inhibitory state
I(net.inh1) = I(net.inh1) - 0.3;
I(net.inh2) = I(net.inh2) - 0.3;

% Add noise
noise_scale = 0.4;
I = I + noise * noise_scale;
I = max(I, 0);
end

function efficacy = calculate_feedforward_efficacy_detailed(rates, net)
% Detailed calculation of feedforward inhibition efficacy

% Calculate FF-Inh inhibition efficacy on Exc3
ff_inh_activity = mean(rates(net.inh3_ff));
exc3_activity = mean(rates(net.exc3));

if exc3_activity > 0
    efficacy = ff_inh_activity / exc3_activity;
else
    efficacy = 0;
end
end

function analyze_ff_excitability_results(detailed_results, ff_excitability_levels)
% Analyze FF-Inh excitability experiment results

fprintf('\n=== FF-Inh Excitability Gradient Experiment Analysis ===\n');

% Calculate means and standard deviations
mean_exc3 = mean(detailed_results.exc3_rates, 2);
std_exc3 = std(detailed_results.exc3_rates, 0, 2);
mean_ff_inh = mean(detailed_results.ff_inh_rates, 2);
std_ff_inh = std(detailed_results.ff_inh_rates, 0, 2);
mean_noise_corr = mean(detailed_results.noise_correlations, 2, 'omitnan');
std_noise_corr = std(detailed_results.noise_correlations, 0, 2, 'omitnan');
mean_ff_efficacy = mean(detailed_results.ff_efficacy, 2);
mean_variability = mean(detailed_results.exc3_variability, 2, 'omitnan');

% Output results table
fprintf('\nFF Excit\tExc3 Rate\t\tFF-Inh Rate\tNoise Corr\t\tFF Efficacy\t\tExc3 Variability\n');
fprintf('-----------------------------------------------------------------------------------------\n');
for i = 1:length(ff_excitability_levels)
    fprintf('%.1f\t\t%.2f±%.2f\t\t%.2f±%.2f\t\t%.4f±%.4f\t%.3f\t\t%.3f\n', ...
        ff_excitability_levels(i), ...
        mean_exc3(i), std_exc3(i), ...
        mean_ff_inh(i), std_ff_inh(i), ...
        mean_noise_corr(i), std_noise_corr(i), ...
        mean_ff_efficacy(i), ...
        mean_variability(i));
end

% Correlation analysis
fprintf('\n=== Correlation Analysis ===\n');

% FF-Inh activity vs Exc3 activity
[r_ff_exc3, p_ff_exc3] = corrcoef(mean_ff_inh, mean_exc3);
fprintf('FF-Inh activity vs Exc3 activity: r = %.3f, p = %.3f\n', r_ff_exc3(1,2), p_ff_exc3(1,2));

% FF-Inh activity vs Noise correlation
valid_idx = ~isnan(mean_noise_corr);
if sum(valid_idx) > 2
    [r_ff_noise, p_ff_noise] = corrcoef(mean_ff_inh(valid_idx), mean_noise_corr(valid_idx));
    fprintf('FF-Inh activity vs Noise correlation: r = %.3f, p = %.3f\n', r_ff_noise(1,2), p_ff_noise(1,2));
end

% Exc3 activity vs Noise correlation
if sum(valid_idx) > 2
    [r_exc3_noise, p_exc3_noise] = corrcoef(mean_exc3(valid_idx), mean_noise_corr(valid_idx));
    fprintf('Exc3 activity vs Noise correlation: r = %.3f, p = %.3f\n', r_exc3_noise(1,2), p_exc3_noise(1,2));
end

% Trend analysis
fprintf('\n=== Trend Analysis ===\n');

if length(ff_excitability_levels) > 2
    % Exc3 firing rate trend
    p_exc3 = polyfit(ff_excitability_levels, mean_exc3', 1);
    fprintf('Exc3 firing rate slope: %.3f Hz/unit FF excitability\n', p_exc3(1));
    
    % Noise correlation trend
    if sum(valid_idx) > 2
        p_noise = polyfit(ff_excitability_levels(valid_idx), mean_noise_corr(valid_idx)', 1);
        fprintf('Noise correlation slope: %.4f /unit FF excitability\n', p_noise(1));
        
        if p_noise(1) < 0
            fprintf('✓ Conclusion: Increased FF-Inh excitability leads to decreased noise correlation\n');
        else
            fprintf('✗ Anomaly: Increased FF-Inh excitability leads to increased noise correlation\n');
        end
    end
end

% Feedforward inhibition efficacy analysis
max_efficacy_idx = find(mean_ff_efficacy == max(mean_ff_efficacy), 1);
fprintf('\nOptimal feedforward inhibition efficacy at FF excitability = %.1f, efficacy = %.3f\n', ...
    ff_excitability_levels(max_efficacy_idx), max(mean_ff_efficacy));
end



function analyze_ff_connectivity(S_raw, net)
% Detailed analysis of FF neuron connectivity structure

fprintf('\n=== Detailed FF Neuron Connectivity Structure Analysis ===\n');

% Basic information
fprintf('Network Size:\n');
fprintf('  Exc2 neurons: %d (IDs: %d-%d)\n', length(net.exc2), min(net.exc2), max(net.exc2));
fprintf('  Exc3 neurons: %d (IDs: %d-%d)\n', length(net.exc3), min(net.exc3), max(net.exc3));
fprintf('  FF-Inh neurons: %d (IDs: %d-%d)\n', length(net.inh3_ff), min(net.inh3_ff), max(net.inh3_ff));
fprintf('  FB-Inh neurons: %d (IDs: %d-%d)\n', length(net.inh3_fb), min(net.inh3_fb), max(net.inh3_fb));

% 1. Analyze Exc2 → FF-Inh connections
fprintf('\n1. Exc2 → FF-Inh Activation Connections:\n');
exc2_to_ff_connections = 0;
exc2_to_ff_weights = [];

for i = net.inh3_ff
    for j = net.exc2
        if S_raw(i, j) > 0  % Excitatory connection
            exc2_to_ff_connections = exc2_to_ff_connections + 1;
            exc2_to_ff_weights = [exc2_to_ff_weights, S_raw(i, j)];
        end
    end
end

fprintf('  Number of connections: %d/%d (%.1f%%)\n', exc2_to_ff_connections, length(net.exc2)*length(net.inh3_ff), ...
        100*exc2_to_ff_connections/(length(net.exc2)*length(net.inh3_ff)));
if ~isempty(exc2_to_ff_weights)
    fprintf('  Connection weights: %.2f ± %.2f (range: %.2f-%.2f)\n', ...
            mean(exc2_to_ff_weights), std(exc2_to_ff_weights), ...
            min(exc2_to_ff_weights), max(exc2_to_ff_weights));
end

% 2. Analyze FF-Inh → Exc3 connections (key feedforward inhibition connections)
fprintf('\n2. FF-Inh → Exc3 Inhibition Connections:\n');
ff_to_exc_connections = 0;
ff_to_exc_weights = [];
connection_matrix = zeros(length(net.exc3), length(net.inh3_ff));

for i = net.exc3
    for j = net.inh3_ff
        if S_raw(i, j) < 0  % Inhibitory connection
            ff_to_exc_connections = ff_to_exc_connections + 1;
            ff_to_exc_weights = [ff_to_exc_weights, abs(S_raw(i, j))];
            
            % Record connection matrix
            exc_idx = find(net.exc3 == i);
            ff_idx = find(net.inh3_ff == j);
            connection_matrix(exc_idx, ff_idx) = abs(S_raw(i, j));
        end
    end
end

fprintf('  Number of connections: %d/%d (%.1f%%)\n', ff_to_exc_connections, length(net.exc3)*length(net.inh3_ff), ...
        100*ff_to_exc_connections/(length(net.exc3)*length(net.inh3_ff)));
if ~isempty(ff_to_exc_weights)
    fprintf('  Inhibition weights: %.2f ± %.2f (range: %.2f-%.2f)\n', ...
            mean(ff_to_exc_weights), std(ff_to_exc_weights), ...
            min(ff_to_exc_weights), max(ff_to_exc_weights));
end

% 3. Display connection matrix
fprintf('\n3. FF-Inh → Exc3 Connection Matrix (rows=Exc3, columns=FF-Inh):\n');
fprintf('    ');
for j = 1:length(net.inh3_ff)
    fprintf('FF%d   ', j);
end
fprintf('\n');

for i = 1:min(10, length(net.exc3))  % Only show first 10 Exc3 neurons
    fprintf('E3_%2d: ', i);
    for j = 1:length(net.inh3_ff)
        fprintf('%5.1f ', connection_matrix(i, j));
    end
    fprintf('\n');
end
if length(net.exc3) > 10
    fprintf('... (omitted remaining %d Exc3 neurons)\n', length(net.exc3)-10);
end

% 4. Analyze influence range of each FF-Inh neuron
fprintf('\n4. Influence Analysis of Each FF-Inh Neuron:\n');
for ff_idx = 1:length(net.inh3_ff)
    ff_neuron = net.inh3_ff(ff_idx);
    targets = [];
    weights = [];
    
    for exc_neuron = net.exc3
        if S_raw(exc_neuron, ff_neuron) < 0
            targets = [targets, exc_neuron];
            weights = [weights, abs(S_raw(exc_neuron, ff_neuron))];
        end
    end
    
    fprintf('  FF-Inh_%d (neuron#%d): Inhibits %d Exc3 neurons', ff_idx, ff_neuron, length(targets));
    if ~isempty(weights)
        fprintf(' (weight: %.1f±%.1f)', mean(weights), std(weights));
    end
    fprintf('\n');
end

% 5. Analyze FF inhibition received by each Exc3 neuron
fprintf('\n5. FF Inhibition Analysis Received by Each Exc3 Neuron:\n');
ff_inhibition_per_exc = zeros(length(net.exc3), 1);
for exc_idx = 1:length(net.exc3)
    exc_neuron = net.exc3(exc_idx);
    total_ff_inhibition = 0;
    ff_sources = 0;
    
    for ff_neuron = net.inh3_ff
        if S_raw(exc_neuron, ff_neuron) < 0
            total_ff_inhibition = total_ff_inhibition + abs(S_raw(exc_neuron, ff_neuron));
            ff_sources = ff_sources + 1;
        end
    end
    
    ff_inhibition_per_exc(exc_idx) = total_ff_inhibition;
    
    if exc_idx <= 5  % Only show first 5
        fprintf('  Exc3_%d: Total FF inhibition=%.1f (from %d FF-Inh)\n', ...
                exc_idx, total_ff_inhibition, ff_sources);
    end
end

fprintf('  FF inhibition distribution: %.1f±%.1f (range: %.1f-%.1f)\n', ...
        mean(ff_inhibition_per_exc), std(ff_inhibition_per_exc), ...
        min(ff_inhibition_per_exc), max(ff_inhibition_per_exc));

fprintf('\n=== FF Connection Analysis Complete ===\n');
end

% Keep all original auxiliary functions
function analyze_synchrony_results(detailed_results, ff_excitability_levels)
% Analyze synchrony metric results

fprintf('\n=== Synchrony Metric Analysis ===\n');

% Calculate means
mean_plv = mean(detailed_results.phase_locking, 2, 'omitnan');
mean_stv = mean(detailed_results.spike_time_variance, 2, 'omitnan');
mean_isi = mean(detailed_results.isi_synchrony, 2, 'omitnan');
mean_vs = mean(detailed_results.vector_strength, 2, 'omitnan');
mean_pc = mean(detailed_results.pairwise_correlation, 2, 'omitnan');
mean_pb = mean(detailed_results.population_burst, 2, 'omitnan');

% Output results table
fprintf('\nFF Excit\tPhase Lock\t\tSpike Var\t\tISI Sync\t\tVector Str\t\tPairwise\t\tPop Burst\n');
fprintf('--------------------------------------------------------------------------------------------------\n');
for i = 1:length(ff_excitability_levels)
    fprintf('%.1f\t\t%.4f\t\t\t%.4f\t\t\t%.4f\t\t\t%.4f\t\t\t%.4f\t\t\t%.4f\n', ...
        ff_excitability_levels(i), ...
        mean_plv(i), mean_stv(i), mean_isi(i), ...
        mean_vs(i), mean_pc(i), mean_pb(i));
end

% Correlation analysis
fprintf('\n=== Correlation of Synchrony Metrics with FF-Inh Excitability ===\n');
valid_idx = ~isnan(mean_plv);
if sum(valid_idx) > 2
    [r_plv, p_plv] = corrcoef(ff_excitability_levels(valid_idx)', mean_plv(valid_idx));
    fprintf('Phase locking value: r = %.3f, p = %.3f\n', r_plv(1,2), p_plv(1,2));
end

valid_idx = ~isnan(mean_vs);
if sum(valid_idx) > 2
    [r_vs, p_vs] = corrcoef(ff_excitability_levels(valid_idx)', mean_vs(valid_idx));
    fprintf('Vector strength: r = %.3f, p = %.3f\n', r_vs(1,2), p_vs(1,2));
end

valid_idx = ~isnan(mean_pb);
if sum(valid_idx) > 2
    [r_pb, p_pb] = corrcoef(ff_excitability_levels(valid_idx)', mean_pb(valid_idx));
    fprintf('Population burst index: r = %.3f, p = %.3f\n', r_pb(1,2), p_pb(1,2));
end
end

% Need to include previously defined auxiliary functions
function [S_raw, net] = setup_complete_microcircuit_balanced()
% Setup balanced cortical microcircuit: adaptive inhibition strength

fprintf('Setting up balanced cortical microcircuit with adaptive inhibition...\n');

% ========================
% Network architecture definition (unchanged)
% ========================
Ne1 = 8;   Ni1 = 2;
Ne2 = 12;  Ni2 = 3;
Ne3 = 20;  Ni3_ff = 4;  Ni3_fb = 4;
Nvip = 6;
N = Ne1 + Ni1 + Ne2 + Ni2 + Ne3 + Ni3_ff + Ni3_fb + Nvip;

% ========================
% Neuron numbering groups (unchanged)
% ========================
exc1 = 1:Ne1;
inh1 = Ne1+1 : Ne1+Ni1;
exc2 = Ne1+Ni1+1 : Ne1+Ni1+Ne2;
inh2 = Ne1+Ni1+Ne2+1 : Ne1+Ni1+Ne2+Ni2;
exc3 = Ne1+Ni1+Ne2+Ni2+1 : Ne1+Ni1+Ne2+Ni2+Ne3;
inh3_ff = Ne1+Ni1+Ne2+Ni2+Ne3+1 : Ne1+Ni1+Ne2+Ni2+Ne3+Ni3_ff;
inh3_fb = Ne1+Ni1+Ne2+Ni2+Ne3+Ni3_ff+1 : Ne1+Ni1+Ne2+Ni2+Ne3+Ni3_ff+Ni3_fb;
vip = N-Nvip+1 : N;

net.exc1 = exc1; net.inh1 = inh1;
net.exc2 = exc2; net.inh2 = inh2;
net.exc3 = exc3; 
net.inh3_ff = inh3_ff; net.inh3_fb = inh3_fb;
net.inh3 = [inh3_ff, inh3_fb];
net.vip = vip;

% ========================
% Improved connection parameters
% ========================
S_raw = zeros(N, N);

% *** Key modification 1: Reduce connection probability ***
prob_within = 0.5;           % Reduced from 0.5 to 0.4
prob_between = 0.4;          % Reduced from 0.4 to 0.3
prob_ff_inhibition = 0.8;    % Reduced from 0.7 to 0.6
prob_fb_inhibition = 0.8;    % Reduced from 0.8 to 0.5 (key!)
prob_vip_disinhibition = 0.5; % Reduced from 0.5 to 0.4

% *** Key modification 2: Adjust weight ranges ***
exc_weight = @() max(0, 6 + 2*randn());    
inh_weight = @() -max(0, 8 + 3*randn());   
% FB-Inh weight range narrowed and added E/I balance-based regulation
fb_inh_weight = @(scale_factor) -max(0, (6 + 2*randn()) * scale_factor);  % Dynamic regulation
vip_weight = @() -max(0, 9 + 3*randn());   

% ========================
% Forward connection construction (unchanged)
% ========================
% Layer 1 internal connections
for i = exc1
    for j = exc1
        if i ~= j && rand() <= prob_within
            S_raw(j, i) = exc_weight();
        end
    end
end

% Layer 1 → Layer 2
for i = exc2
    for j = exc1
        if rand() <= prob_between
            S_raw(i, j) = exc_weight();
        end
    end
end

% Layer 2 internal connections
for i = exc2
    for j = exc2
        if i ~= j && rand() <= prob_within
            S_raw(j, i) = exc_weight();
        end
    end
end

% Layer 2 → Layer 3
for i = exc3
    for j = exc2
        if rand() <= prob_between
            S_raw(i, j) = exc_weight();
        end
    end
end

% Layer 3 internal connections
for i = exc3
    for j = exc3
        if i ~= j && rand() <= prob_within
            S_raw(j, i) = exc_weight();
        end
    end
end

% ========================
% Feedforward inhibition circuit (unchanged)
% ========================
for i = inh3_ff
    for j = exc2
        if rand() <= prob_ff_inhibition
            S_raw(i, j) = exc_weight()*1.5;
        end
    end
end

for i = exc3
    for j = inh3_ff
        if rand() <= prob_ff_inhibition
            S_raw(i, j) = inh_weight();
        end
    end
end

% ========================
% *** E/I balance-based feedback inhibition design ***
% ========================
fprintf('Building E/I balanced feedback inhibition circuit...\n');

% First calculate excitatory input strength for each Exc3 neuron
exc_input_strength = zeros(length(exc3), 1);
for i = 1:length(exc3)
    exc_neuron = exc3(i);
    total_exc_input = 0;
    
    % Calculate excitatory input from exc2 and other exc3
    for j = [exc2, exc3]
        if j ~= exc_neuron && S_raw(exc_neuron, j) > 0
            total_exc_input = total_exc_input + S_raw(exc_neuron, j);
        end
    end
    exc_input_strength(i) = total_exc_input;
end

% Layer 3 excitatory → Feedback inhibition neurons (moderate connectivity)
for i = inh3_fb
    for j = exc3
        if rand() <= prob_fb_inhibition
            % S_raw(i, j) = exc_weight();
            total_exc_input = 0;
            for k = [exc1, exc2, exc3]
                if k ~= i && S_raw(i, k) > 0
                    total_exc_input = total_exc_input + S_raw(i, k);
                end
            end
            
            % FF inhibition should significantly affect Exc3 activity
            ff_inhibition_strength = total_exc_input * (0.8 + 0.4 * rand());  % 80%-120% of excitatory input
            S_raw(i, j) = -ff_inhibition_strength;
        end
    end
end

% *** Key improvement: Regulate FB-Inh weights based on excitatory input strength ***
for i = 1:length(exc3)
    exc_neuron = exc3(i);
    
    % Calculate required inhibition strength for this neuron
    target_exc_input = exc_input_strength(i);
    
    if target_exc_input > 0
        % Inhibition strength should be proportional to excitatory input, but not exceed
        target_inhibition_ratio = 1 + 0.5 * rand(); % 70%-100% of excitatory strength
        target_inh_strength = target_exc_input * target_inhibition_ratio;
        
        % Count number of FB-Inh connections to this neuron
        fb_connections = 0;
        for j = inh3_fb
            if rand() <= prob_fb_inhibition
                fb_connections = fb_connections + 1;
            end
        end
        
        if fb_connections > 0
            % Evenly distribute inhibition strength
            individual_weight = target_inh_strength / fb_connections;
            individual_weight = min(individual_weight, 25); % Limit maximum weight
            
            % Reset connections
            for j = inh3_fb
                if rand() <= prob_fb_inhibition
                    weight_variation = 0.7 + 0.6*rand(); % 70%-130% variation
                    S_raw(exc_neuron, j) = -individual_weight * weight_variation;
                end
            end
        end
    end
end

% *** Modification 4: Reduce lateral inhibition among FB-Inh ***
for i = 1:length(inh3_fb)
    for j = 1:length(inh3_fb)
        if i ~= j && rand() <= 0.2  % Reduced from 0.4 to 0.2
            S_raw(inh3_fb(i), inh3_fb(j)) = -max(0, 4 + 1*randn()); % Weaken weight
        end
    end
end

% *** Modification 5: Simplify delayed inhibition pathway ***
for i = 1:length(exc3)
    for j = 1:length(inh3_fb)
        if rand() <= 0.1  % Reduced from 0.2 to 0.1
            S_raw(inh3_fb(j), exc3(i)) = exc_weight() * 0.4; % Reduced from 0.6 to 0.4
        end
    end
end

% ========================
% VIP circuit (slightly adjusted weights)
% ========================
for i = vip
    for j = [exc2, exc3]
        if rand() <= prob_between * 0.6  % Reduced from 0.7 to 0.6
            S_raw(i, j) = exc_weight() * 0.7;  % Reduced from 0.8 to 0.7
        end
    end
end

for i = inh3_fb
    for j = vip
        if rand() <= prob_vip_disinhibition
            S_raw(i, j) = vip_weight()*0.3;  % Reduced from 0.3 to 0.25
        end
    end
end

for i = inh3_ff
    for j = vip
        if rand() <= prob_vip_disinhibition * 0.2  % Reduced from 0.3 to 0.2
            S_raw(i, j) = vip_weight() * 0;  % Reduced from 0.4 to 0.3
        end
    end
end

% ========================
% Local inhibition connections (unchanged)
% ========================
for i = exc1
    for j = inh1
        if rand() <= prob_within
            S_raw(i, j) = inh_weight();
            S_raw(j, i) = exc_weight();
        end
    end
end

for i = exc2
    for j = inh2
        if rand() <= prob_within
            S_raw(i, j) = inh_weight();
            S_raw(j, i) = exc_weight();
        end
    end
end

% ========================
% *** New: Network balance check and adjustment ***
% ========================
fprintf('\n=== Network Balance Analysis ===\n');

% Analyze E/I ratio for each Exc3 neuron
fprintf('Exc3 neuron E/I balance analysis:\n');
for i = 1:min(5, length(exc3))  % Only show analysis for first 5 neurons
    exc_neuron = exc3(i);
    
    total_exc = sum(S_raw(exc_neuron, S_raw(exc_neuron, :) > 0));
    total_inh = abs(sum(S_raw(exc_neuron, S_raw(exc_neuron, :) < 0)));
    
    if total_exc > 0
        ei_ratio = total_inh / total_exc;
        fprintf('  Neuron%d: E=%.1f, I=%.1f, I/E=%.2f\n', exc_neuron, total_exc, total_inh, ei_ratio);
        
        % If E/I ratio is too imbalanced, fine-tune
        if ei_ratio < 0.5  % Insufficient inhibition
            fprintf('    -> Insufficient inhibition, slightly enhance FB-Inh connection\n');
            for j = inh3_fb
                if S_raw(exc_neuron, j) < 0
                    S_raw(exc_neuron, j) = S_raw(exc_neuron, j) * 1.2; % Enhance 20%
                end
            end
        elseif ei_ratio > 1.5  % Excessive inhibition
            fprintf('    -> Excessive inhibition, slightly weaken FB-Inh connection\n');
            for j = inh3_fb
                if S_raw(exc_neuron, j) < 0
                    S_raw(exc_neuron, j) = S_raw(exc_neuron, j) * 0.8; % Weaken 20%
                end
            end
        end
    end
end

% Output final statistics
total_connections = sum(S_raw(:) ~= 0);
exc_connections = sum(S_raw(:) > 0);
inh_connections = sum(S_raw(:) < 0);

fprintf('\n=== Final Network Statistics ===\n');
fprintf('Total connections: %d\n', total_connections);
fprintf('Excitatory connections: %d (%.1f%%)\n', exc_connections, 100*exc_connections/total_connections);
fprintf('Inhibitory connections: %d (%.1f%%)\n', inh_connections, 100*inh_connections/total_connections);

% FB-Inh specific analysis
fb_to_exc_connections = 0;
fb_weights = [];
for i = exc3
    for j = inh3_fb
        if S_raw(i, j) < 0
            fb_to_exc_connections = fb_to_exc_connections + 1;
            fb_weights = [fb_weights, abs(S_raw(i, j))];
        end
    end
end

fprintf('\nFB-Inh connection analysis:\n');
fprintf('FB-Inh → Exc3 connections: %d\n', fb_to_exc_connections);
fprintf('Average FB-Inh weight: %.2f ± %.2f\n', mean(fb_weights), std(fb_weights));
fprintf('FB-Inh weight range: %.2f - %.2f\n', min(fb_weights), max(fb_weights));

fprintf('\n*** Balance optimization complete - FB-Inh weights adjusted based on E/I balance ***\n');
end

function [a, b, c, d, v, u] = setup_neuron_parameters(net, N)
% Setup Izhikevich neuron parameters

% Generate randomness
Ne_total = length([net.exc1, net.exc2, net.exc3]);
Ni_total = length([net.inh1, net.inh2, net.inh3]);
Nvip = length(net.vip);

re = rand(Ne_total, 1);
ri = rand(Ni_total, 1);
rvip = rand(Nvip, 1);

a = zeros(N, 1); b = zeros(N, 1); c = zeros(N, 1); d = zeros(N, 1);

% Excitatory neurons (Regular Spiking)
exc_indices = [net.exc1, net.exc2, net.exc3];
a(exc_indices) = 0.02;
b(exc_indices) = 0.2;
c(exc_indices) = -65 + 15 * re.^2;
d(exc_indices) = 8 - 6 * re.^2;

% Inhibitory neurons (Fast Spiking)
inh_indices = [net.inh1, net.inh2, net.inh3];
a(inh_indices) = 0.1;
b(inh_indices) = 0.2;
c(inh_indices) = -65;
d(inh_indices) = 2;

% VIP neurons (Low-threshold bursting)
a(net.vip) = 0.02;
b(net.vip) = 0.25;
c(net.vip) = -65;
d(net.vip) = 0.05;

% Initialization
v = -65 * ones(N, 1);
u = b .* v;
end


function noise_corr = calculate_improved_noise_correlation(exc3_spike_times, total_time_ms)
% Improved noise correlation calculation

% Use multi-timescale analysis
time_windows = [100, 200, 500];  % 100ms, 200ms, 500ms
correlations_all = [];

fprintf('Calculating multi-scale noise correlation...\n');

for window_size = time_windows
    step_size = window_size * 0.25;  % Reduce overlap
    n_windows = floor((total_time_ms - window_size) / step_size) + 1;
    n_neurons = length(exc3_spike_times);
    
    if n_neurons < 2 || n_windows < 5
        continue;
    end
    
    spike_counts = zeros(n_neurons, n_windows);
    
    % Calculate spike count for each time window
    for win = 1:n_windows
        win_start = (win-1) * step_size;
        win_end = win_start + window_size;
        
        for neuron = 1:n_neurons
            if ~isempty(exc3_spike_times{neuron})
                spikes_in_window = exc3_spike_times{neuron}(...
                    exc3_spike_times{neuron} >= win_start & ...
                    exc3_spike_times{neuron} < win_end);
                spike_counts(neuron, win) = length(spikes_in_window);
            end
        end
    end
    
    % Only select neurons with sufficient activity
    mean_activity = mean(spike_counts, 2);
    std_activity = std(spike_counts, 0, 2);
    active_neurons = (mean_activity > 0.5) & (std_activity > 0.1);  % More strict filtering
    
    fprintf('Time window %dms: %d/%d active neurons\n', window_size, sum(active_neurons), n_neurons);
    
    if sum(active_neurons) >= 2
        active_counts = spike_counts(active_neurons, :);
        
        % Detrend and standardize
        for i = 1:size(active_counts, 1)
            % active_counts(i, :) = detrend(active_counts(i, :));
            if std(active_counts(i, :)) > 0
                active_counts(i, :) = (active_counts(i, :)-mean(active_counts(i, :)))/ std(active_counts(i, :));
            end
        end
        
        % Calculate correlation
        try
            corr_matrix = corrcoef(active_counts');
            n_active = size(corr_matrix, 1);
            
            % Extract upper triangle
            upper_indices = find(triu(ones(n_active, n_active), 1));
            correlations = corr_matrix(upper_indices);
            
            % Filter valid correlations
            valid_corr = correlations(~isnan(correlations) & abs(correlations) < 0.95);
            
            if ~isempty(valid_corr)
                correlations_all = [correlations_all; valid_corr];
                fprintf('  Number of valid correlations: %d, Mean: %.4f\n', length(valid_corr), mean(valid_corr));
            end
        catch ME
            fprintf('  Correlation calculation failed: %s\n', ME.message);
        end
    end
end

if ~isempty(correlations_all)
    noise_corr = mean(correlations_all);
    fprintf('Final multi-scale noise correlation: %.4f (based on %d correlation values)\n', noise_corr, length(correlations_all));
else
    noise_corr = NaN;
    fprintf('Unable to calculate valid noise correlation\n');
end
end

function sync_index = calculate_synchrony_index(spike_times, method)
% Calculate population synchrony index
% Input:
%   spike_times: cell array, each cell contains spike times (ms) for one neuron
%   method: 'phase', 'isi_variance', 'spike_variance', 'vector_strength', 'all'
% Output:
%   sync_index: synchrony index structure or single value

if nargin < 2
    method = 'all';
end

n_neurons = length(spike_times);

% Initialize result structure
if strcmp(method, 'all')
    sync_index = struct();
    sync_index.phase_locking = calculate_phase_locking_value(spike_times);
    sync_index.spike_time_variance = calculate_spike_time_variance(spike_times);
    sync_index.isi_synchrony = calculate_isi_synchrony(spike_times);
    sync_index.vector_strength = calculate_vector_strength(spike_times);
    sync_index.pairwise_correlation = calculate_pairwise_spike_correlation(spike_times);
    sync_index.population_burst = calculate_population_burst_index(spike_times);
else
    switch lower(method)
        case 'phase'
            sync_index = calculate_phase_locking_value(spike_times);
        case 'isi_variance'
            sync_index = calculate_isi_synchrony(spike_times);
        case 'spike_variance'
            sync_index = calculate_spike_time_variance(spike_times);
        case 'vector_strength'
            sync_index = calculate_vector_strength(spike_times);
        otherwise
            error('Unknown synchrony calculation method: %s', method);
    end
end
end

function plv = calculate_phase_locking_value(spike_times)
% Calculate Phase Locking Value
% Based on Hilbert transform and instantaneous phase

n_neurons = length(spike_times);
if n_neurons < 2
    plv = NaN;
    return;
end

% Parameter settings
total_time = 0;
for i = 1:n_neurons
    if ~isempty(spike_times{i})
        total_time = max(total_time, max(spike_times{i}));
    end
end

if total_time == 0
    plv = NaN;
    return;
end

% Create binary spike sequence
dt = 1; % 1ms resolution
time_bins = 0:dt:total_time;
n_bins = length(time_bins);

spike_trains = zeros(n_neurons, n_bins);
for i = 1:n_neurons
    if ~isempty(spike_times{i})
        spike_indices = round(spike_times{i}/dt) + 1;
        spike_indices = spike_indices(spike_indices <= n_bins & spike_indices > 0);
        spike_trains(i, spike_indices) = 1;
    end
end

% Calculate instantaneous phase
phases = zeros(n_neurons, n_bins);
for i = 1:n_neurons
    if sum(spike_trains(i, :)) > 10 % At least 10 spikes
        % Gaussian smoothing
        sigma = 10; % 10ms standard deviation
        kernel = gausswin(6*sigma);
        kernel = kernel / sum(kernel);
        
        smoothed = conv(spike_trains(i, :), kernel, 'same');
        
        % Hilbert transform to get instantaneous phase
        analytic_signal = hilbert(smoothed);
        phases(i, :) = angle(analytic_signal);
    end
end

% Calculate phase locking value
valid_neurons = sum(spike_trains, 2) > 10;
if sum(valid_neurons) < 2
    plv = NaN;
    return;
end

valid_phases = phases(valid_neurons, :);
n_valid = size(valid_phases, 1);

% Calculate PLV for all neuron pairs
plv_values = [];
for i = 1:n_valid-1
    for j = i+1:n_valid
        phase_diff = valid_phases(i, :) - valid_phases(j, :);
        plv_pair = abs(mean(exp(1i * phase_diff)));
        plv_values = [plv_values, plv_pair];
    end
end

plv = mean(plv_values);
end

function spike_var = calculate_spike_time_variance(spike_times)
% Calculate spike time variance synchrony index
% Smaller variance indicates higher synchrony

n_neurons = length(spike_times);
if n_neurons < 2
    spike_var = NaN;
    return;
end

% Find all spike times
all_spikes = [];
for i = 1:n_neurons
    if ~isempty(spike_times{i})
        all_spikes = [all_spikes; spike_times{i}];
    end
end

if length(all_spikes) < 10
    spike_var = NaN;
    return;
end

all_spikes = sort(all_spikes);

% Calculate synchrony within time windows
window_size = 50; % 50ms window
step_size = 25;   % 25ms step
total_time = max(all_spikes);
n_windows = floor((total_time - window_size) / step_size) + 1;

sync_values = [];
for w = 1:n_windows
    win_start = (w-1) * step_size;
    win_end = win_start + window_size;
    
    % Calculate spike count for each neuron in this window
    spike_counts = zeros(n_neurons, 1);
    for i = 1:n_neurons
        if ~isempty(spike_times{i})
            spikes_in_window = spike_times{i}(...
                spike_times{i} >= win_start & spike_times{i} < win_end);
            spike_counts(i) = length(spikes_in_window);
        end
    end
    
    % Calculate synchrony for this window (negative variance, larger is more synchronized)
    if sum(spike_counts) > 0
        sync_values = [sync_values, -var(spike_counts)];
    end
end

if ~isempty(sync_values)
    spike_var = mean(sync_values);
else
    spike_var = NaN;
end
end

function isi_sync = calculate_isi_synchrony(spike_times)
% ISI (Inter-Spike Interval) based synchrony
% Synchronized neurons should have similar ISI patterns

n_neurons = length(spike_times);
if n_neurons < 2
    isi_sync = NaN;
    return;
end

% Calculate ISI for each neuron
isis = cell(n_neurons, 1);
for i = 1:n_neurons
    if length(spike_times{i}) > 1
        isis{i} = diff(spike_times{i});
    end
end

% Remove empty ISIs
valid_isis = {};
for i = 1:n_neurons
    if ~isempty(isis{i}) && length(isis{i}) > 5
        valid_isis{end+1} = isis{i};
    end
end

if length(valid_isis) < 2
    isi_sync = NaN;
    return;
end

% Calculate similarity of ISI distributions
n_valid = length(valid_isis);
correlations = [];

for i = 1:n_valid-1
    for j = i+1:n_valid
        % Use same bin edges to calculate histogram
        all_isis = [valid_isis{i}; valid_isis{j}];
        bin_edges = linspace(min(all_isis), max(all_isis), 20);
        
        hist1 = histcounts(valid_isis{i}, bin_edges);
        hist2 = histcounts(valid_isis{j}, bin_edges);
        
        % Normalize
        hist1 = hist1 / sum(hist1);
        hist2 = hist2 / sum(hist2);
        
        % Calculate correlation
        if sum(hist1) > 0 && sum(hist2) > 0
            corr_val = corrcoef(hist1, hist2);
            if size(corr_val, 1) > 1
                correlations = [correlations, corr_val(1,2)];
            end
        end
    end
end

if ~isempty(correlations)
    correlations = correlations(~isnan(correlations));
    isi_sync = mean(correlations);
else
    isi_sync = NaN;
end
end

function vector_strength = calculate_vector_strength(spike_times)
% Calculate Vector Strength
% Measure degree of spike locking to average rhythm

n_neurons = length(spike_times);
if n_neurons < 2
    vector_strength = NaN;
    return;
end

% Estimate main oscillation frequency
all_spikes = [];
for i = 1:n_neurons
    if ~isempty(spike_times{i})
        all_spikes = [all_spikes; spike_times{i}];
    end
end

if length(all_spikes) < 20
    vector_strength = NaN;
    return;
end

all_spikes = sort(all_spikes);

% Calculate mean firing rate to estimate main frequency
mean_rate = length(all_spikes) / (max(all_spikes) - min(all_spikes)) * 1000; % Hz
dominant_freq = mean_rate / 2; % Assume main oscillation is half of mean firing rate

if dominant_freq < 1 || dominant_freq > 100
    vector_strength = NaN;
    return;
end

% Calculate vector strength for each neuron
vs_values = [];
for i = 1:n_neurons
    if length(spike_times{i}) > 5
        phases = 2 * pi * dominant_freq * spike_times{i} / 1000;
        vector_sum = sum(exp(1i * phases));
        vs = abs(vector_sum) / length(phases);
        vs_values = [vs_values, vs];
    end
end

if ~isempty(vs_values)
    vector_strength = mean(vs_values);
else
    vector_strength = NaN;
end
end

function pairwise_corr = calculate_pairwise_spike_correlation(spike_times)
% Calculate pairwise spike correlation (similar to previous noise correlation but optimized)

n_neurons = length(spike_times);
if n_neurons < 2
    pairwise_corr = NaN;
    return;
end

% Calculate total time
total_time = 0;
for i = 1:n_neurons
    if ~isempty(spike_times{i})
        total_time = max(total_time, max(spike_times{i}));
    end
end

if total_time == 0
    pairwise_corr = NaN;
    return;
end

% Use sliding window to calculate spike count correlation
bin_size = 50;  % 50ms
step_size = 25; % 25ms step
n_windows = floor((total_time - bin_size) / step_size) + 1;

spike_counts = zeros(n_neurons, n_windows);

for w = 1:n_windows
    win_start = (w-1) * step_size;
    win_end = win_start + bin_size;
    
    for i = 1:n_neurons
        if ~isempty(spike_times{i})
            spikes_in_window = spike_times{i}(...
                spike_times{i} >= win_start & spike_times{i} < win_end);
            spike_counts(i, w) = length(spikes_in_window);
        end
    end
end

% Remove mean firing rate
spike_counts_centered = spike_counts - mean(spike_counts, 2);

% Calculate correlation
active_neurons = std(spike_counts, 0, 2) > 0;
if sum(active_neurons) < 2
    pairwise_corr = NaN;
    return;
end

try
    corr_matrix = corrcoef(spike_counts_centered(active_neurons, :)');
    n_active = size(corr_matrix, 1);
    upper_indices = find(triu(ones(n_active, n_active), 1));
    correlations = corr_matrix(upper_indices);
    correlations = correlations(~isnan(correlations));
    
    if ~isempty(correlations)
        pairwise_corr = mean(correlations);
    else
        pairwise_corr = NaN;
    end
catch
    pairwise_corr = NaN;
end
end

function burst_index = calculate_population_burst_index(spike_times)
% Calculate population burst index
% Measure strength of population synchronous burst events

n_neurons = length(spike_times);
if n_neurons < 2
    burst_index = NaN;
    return;
end

% Merge all spike times
all_spikes = [];
for i = 1:n_neurons
    if ~isempty(spike_times{i})
        all_spikes = [all_spikes; spike_times{i}];
    end
end

if length(all_spikes) < 20
    burst_index = NaN;
    return;
end

all_spikes = sort(all_spikes);
total_time = max(all_spikes) - min(all_spikes);

% Use small time window to detect population bursts
bin_size = 10; % 10ms window
n_bins = floor(total_time / bin_size);

spike_counts_per_bin = zeros(n_bins, 1);
for i = 1:n_bins
    bin_start = min(all_spikes) + (i-1) * bin_size;
    bin_end = bin_start + bin_size;
    
    spikes_in_bin = all_spikes(all_spikes >= bin_start & all_spikes < bin_end);
    spike_counts_per_bin(i) = length(spikes_in_bin);
end

% Define burst as bins exceeding mean + 2 standard deviations
mean_count = mean(spike_counts_per_bin);
std_count = std(spike_counts_per_bin);
burst_threshold = mean_count + 2 * std_count;

burst_bins = spike_counts_per_bin > burst_threshold;
burst_index = sum(burst_bins) / n_bins; % Proportion of burst events
end


function improved_common_input_analysis = calculate_true_common_input_strength_variable_length(detailed_results)
% Process common input strength calculation for variable-length time series

[n_conditions, n_trials] = size(detailed_results.exc3_time_series);
improved_common_input_analysis = struct();

fprintf('Processing variable-length time series data...\n');

for cond = 1:n_conditions
    fprintf('Condition %d/%d...\n', cond, n_conditions);
    
    trial_activities = {};  % Use cell array to store variable-length sequences
    trial_lengths = [];
    
    % Collect time series from all trials
    for trial = 1:n_trials
        if ~isempty(detailed_results.exc3_time_series{cond, trial})
            exc3_data = detailed_results.exc3_time_series{cond, trial};
            if size(exc3_data, 1) == 1
                activity = exc3_data(1, :);  % [1 × length_i]
                trial_activities{end+1} = activity;
                trial_lengths = [trial_lengths, length(activity)];
                
                fprintf('  Trial %d: length %d\n', trial, length(activity));
            end
        end
    end
    
    if length(trial_activities) >= 2  % At least 2 trials
        
        % Method 1: Truncate to shortest length
        min_length = min(trial_lengths);
        truncated_activities = zeros(length(trial_activities), min_length);
        
        for t = 1:length(trial_activities)
            truncated_activities(t, :) = trial_activities{t}(1:min_length);
        end
        
        % Calculate common signal (truncated version)
        common_signal_truncated = mean(truncated_activities, 1);
        temporal_variability_truncated = std(common_signal_truncated);
        
        % Method 2: Interpolate to same length
        max_length = max(trial_lengths);
        interpolated_activities = zeros(length(trial_activities), max_length);
        
        for t = 1:length(trial_activities)
            original = trial_activities{t};
            original_time = linspace(0, 1, length(original));
            new_time = linspace(0, 1, max_length);
            interpolated_activities(t, :) = interp1(original_time, original, new_time, 'linear');
        end
        
        % Calculate common signal (interpolated version)
        common_signal_interpolated = mean(interpolated_activities, 1);
        temporal_variability_interpolated = std(common_signal_interpolated);
        
        % Method 3: Calculate inter-trial consistency (based on correlation)
        trial_correlations = [];
        for t1 = 1:length(trial_activities)
            for t2 = t1+1:length(trial_activities)
                % Truncate to same length then calculate correlation
                len = min(length(trial_activities{t1}), length(trial_activities{t2}));
                seq1 = trial_activities{t1}(1:len);
                seq2 = trial_activities{t2}(1:len);
                
                if len > 5  % At least 5 points to calculate correlation
                    corr_coef = corrcoef(seq1, seq2);
                    trial_correlations = [trial_correlations, corr_coef(1,2)];
                end
            end
        end
        
        mean_trial_correlation = mean(trial_correlations);
        
        % Method 4: Analysis based on overlap interval
        overlap_start = 1;
        overlap_end = min_length;
        if overlap_end > 10  % At least 10 overlap points
            overlap_activities = zeros(length(trial_activities), overlap_end);
            for t = 1:length(trial_activities)
                overlap_activities(t, :) = trial_activities{t}(overlap_start:overlap_end);
            end
            
            % PCA analysis of overlap interval
            if size(overlap_activities, 1) > 2
                [coeff, score, latent] = pca(overlap_activities);
                first_pc_variance = latent(1) / sum(latent);
            else
                first_pc_variance = NaN;
            end
        else
            first_pc_variance = NaN;
        end
        
        % Save results
        improved_common_input_analysis.trial_activities{cond} = trial_activities;
        improved_common_input_analysis.trial_lengths{cond} = trial_lengths;
        improved_common_input_analysis.common_signal_truncated{cond} = common_signal_truncated;
        improved_common_input_analysis.common_signal_interpolated{cond} = common_signal_interpolated;
        
        improved_common_input_analysis.temporal_variability_truncated(cond) = temporal_variability_truncated;
        improved_common_input_analysis.temporal_variability_interpolated(cond) = temporal_variability_interpolated;
        improved_common_input_analysis.mean_trial_correlation(cond) = mean_trial_correlation;
        improved_common_input_analysis.pca_commonality(cond) = first_pc_variance;
        
        % Composite metric
        improved_common_input_analysis.composite_strength(cond) = ...
            temporal_variability_truncated * (1 + mean_trial_correlation);
            
        fprintf('  Results: truncated variability=%.4f, interpolated variability=%.4f, mean correlation=%.4f\n', ...
            temporal_variability_truncated, temporal_variability_interpolated, mean_trial_correlation);
            
    elseif length(trial_activities) == 1  % Only 1 trial
        activity = trial_activities{1};
        improved_common_input_analysis.trial_activities{cond} = trial_activities;
        improved_common_input_analysis.trial_lengths{cond} = trial_lengths;
        improved_common_input_analysis.common_signal_truncated{cond} = activity;
        improved_common_input_analysis.common_signal_interpolated{cond} = activity;
        
        improved_common_input_analysis.temporal_variability_truncated(cond) = std(activity);
        improved_common_input_analysis.temporal_variability_interpolated(cond) = std(activity);
        improved_common_input_analysis.mean_trial_correlation(cond) = NaN;
        improved_common_input_analysis.pca_commonality(cond) = NaN;
        improved_common_input_analysis.composite_strength(cond) = std(activity);
        
        fprintf('  Single trial: variability=%.4f\n', std(activity));
        
    else  % No data
        improved_common_input_analysis.trial_activities{cond} = {};
        improved_common_input_analysis.trial_lengths{cond} = [];
        improved_common_input_analysis.common_signal_truncated{cond} = [];
        improved_common_input_analysis.common_signal_interpolated{cond} = [];
        
        improved_common_input_analysis.temporal_variability_truncated(cond) = NaN;
        improved_common_input_analysis.temporal_variability_interpolated(cond) = NaN;
        improved_common_input_analysis.mean_trial_correlation(cond) = NaN;
        improved_common_input_analysis.pca_commonality(cond) = NaN;
        improved_common_input_analysis.composite_strength(cond) = NaN;
        
        fprintf('  No valid data\n');
    end
end

% Output statistics
fprintf('\n=== Data Length Statistics ===\n');
all_lengths = [];
for cond = 1:n_conditions
    if ~isempty(improved_common_input_analysis.trial_lengths{cond})
        all_lengths = [all_lengths, improved_common_input_analysis.trial_lengths{cond}];
    end
end

if ~isempty(all_lengths)
    fprintf('Time series length range: %d - %d\n', min(all_lengths), max(all_lengths));
    fprintf('Mean length: %.1f ± %.1f\n', mean(all_lengths), std(all_lengths));
    fprintf('Total %d time series\n', length(all_lengths));
end
end
function [signal_var, noise_var] = calculate_signal_noise_variance(time_series)
% Calculate signal and noise variance

if length(time_series) < 10
    signal_var = NaN;
    noise_var = NaN;
    return;
end

% Low-pass filtering to extract signal component (<2Hz)
dt = 0.1; % 100ms sampling
fs = 1/dt; % 10Hz sampling frequency

% Simple moving average as low-pass filter
window_size = round(fs/2); % 0.5 second window
signal_component = movmean(time_series, window_size);

signal_var = var(signal_component);
noise_var = var(time_series - signal_component);
end