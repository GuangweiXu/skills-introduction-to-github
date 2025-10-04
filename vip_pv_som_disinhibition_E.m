% vip_disinhibition_noise_correlation_analysis.m
clear; clc; close all

% This network model was constructed based on Izhikevich neural network
% Create by Guangwei Xu

% =======================
% Study of VIP disinhibition regulation on Exc3 noise correlation
% =======================


% create neural network
[S_raw, net] = setup_complete_microcircuit_balanced();
N = size(S_raw, 1);
% visualize_vip_microcircuit_network(S_raw, net)
% Define the VIP activation strength gradient
vip_activation_levels = 1:0.1:2.5;  
num_repeats = 50;

fprintf('Start a VIP study to systematically investigate the regulation related to noise....\n');
fprintf('VIP Activation Gradient: %.1f to %.1f, total %d levels\n', ...
    min(vip_activation_levels), max(vip_activation_levels), length(vip_activation_levels));
fprintf('Each condition repeated %d times\n', num_repeats);

% Run VIP disinhibition gradient experiment
detailed_results = run_vip_disinhibition_experiment(S_raw, net, vip_activation_levels, num_repeats);

% Analyze and visualize results
analyze_vip_disinhibition_results(detailed_results, vip_activation_levels, net);
create_vip_disinhibition_comprehensive_figure(detailed_results, vip_activation_levels, net);
analyze_vip_mechanism_details(detailed_results, vip_activation_levels, net);
create_vip_mechanism_figure(detailed_results, vip_activation_levels);

% =======================
% Common input analysis
% =======================

improved_results = calculate_true_common_input_strength_variable_length_vip(detailed_results);

% Visualize improved common input analysis - add error bars
figure('Position', [100, 100, 1400, 800]);

% Calculate error bar data
temporal_var_truncated_mean = improved_results.temporal_variability_truncated;
temporal_var_truncated_std = calculate_std_across_trials(detailed_results, 'temporal_variability_truncated');

temporal_var_interpolated_mean = improved_results.temporal_variability_interpolated;
temporal_var_interpolated_std = calculate_std_across_trials(detailed_results, 'temporal_variability_interpolated');

trial_correlation_mean = improved_results.mean_trial_correlation;
trial_correlation_std = calculate_std_across_trials(detailed_results, 'trial_correlation');

pca_commonality_mean = improved_results.pca_commonality;
pca_commonality_std = calculate_std_across_trials(detailed_results, 'pca_commonality');

composite_strength_mean = improved_results.composite_strength;
composite_strength_std = calculate_std_across_trials(detailed_results, 'composite_strength');

lengths_per_condition = cellfun(@(x) mean(x), improved_results.trial_lengths);
lengths_std = cellfun(@(x) std(x), improved_results.trial_lengths);

subplot(2, 3, 1);
errorbar(vip_activation_levels, temporal_var_truncated_mean, temporal_var_truncated_std, 'o-', 'LineWidth', 2);
xlabel('VIP Activation Level');
ylabel('Common Input Strength (Truncation Method)');
title('A. Temporal Variability - Truncation Method');
grid on;

subplot(2, 3, 2);
errorbar(vip_activation_levels, temporal_var_interpolated_mean, temporal_var_interpolated_std, 's-', 'LineWidth', 2);
xlabel('VIP Activation Level');
ylabel('Common Input Strength (Interpolation Method)');
title('B. Temporal Variability - Interpolation Method');
grid on;

subplot(2, 3, 3);
errorbar(vip_activation_levels, trial_correlation_mean, trial_correlation_std, '^-', 'LineWidth', 2);
xlabel('VIP Activation Level');
ylabel('Mean Inter-trial Correlation');
title('C. Inter-trial Consistency');
grid on;

subplot(2, 3, 4);
errorbar(vip_activation_levels, pca_commonality_mean, pca_commonality_std, 'd-', 'LineWidth', 2);
xlabel('VIP Activation Level');
ylabel('First Principal Component Contribution');
title('D. PCA Commonality');
grid on;

subplot(2, 3, 5);
errorbar(vip_activation_levels, composite_strength_mean, composite_strength_std, 'p-', 'LineWidth', 2);
xlabel('VIP Activation Level');
ylabel('Composite Common Input Strength');
title('E. Composite Metric');
grid on;

% Display sequence length information
subplot(2, 3, 6);
errorbar(vip_activation_levels, lengths_per_condition, lengths_std, 'v-', 'LineWidth', 2);
xlabel('VIP Activation Level');
ylabel('Mean Sequence Length');
title('F. Time Series Length');
grid on;

sgtitle('Analysis of VIP Activation Effects on Common Input with Variable-Length Time Series', 'FontSize', 16);

% =======================
% Core function definitions
% =======================

function std_values = calculate_std_across_trials(detailed_results, metric_name)
% Calculate standard deviation across trials
n_conditions = size(detailed_results.exc3_rates, 1);
std_values = zeros(n_conditions, 1);

% Use standard deviation of noise correlation as proxy
for i = 1:n_conditions
    std_values(i) = std(detailed_results.noise_correlations(i, :), 'omitnan') * 0.5; % Scaling factor
end
end

function detailed_results = run_vip_disinhibition_experiment(S_raw, net, vip_activation_levels, num_repeats)
% Run VIP disinhibition gradient experiment

n_conditions = length(vip_activation_levels);
detailed_results = struct();

% Basic neural activity metrics
detailed_results.exc3_rates = zeros(n_conditions, num_repeats);
detailed_results.vip_rates = zeros(n_conditions, num_repeats);
detailed_results.fb_inh_rates = zeros(n_conditions, num_repeats);
detailed_results.ff_inh_rates = zeros(n_conditions, num_repeats);

% Key noise correlation metrics
detailed_results.noise_correlations = zeros(n_conditions, num_repeats);
detailed_results.pairwise_correlations = zeros(n_conditions, num_repeats);
detailed_results.correlation_distribution_width = zeros(n_conditions, num_repeats);

% VIP disinhibition mechanism metrics
detailed_results.vip_to_fb_efficacy = zeros(n_conditions, num_repeats);
detailed_results.fb_suppression_index = zeros(n_conditions, num_repeats);
detailed_results.disinhibition_index = zeros(n_conditions, num_repeats);
detailed_results.exc3_variability = zeros(n_conditions, num_repeats);

% Network dynamics metrics
detailed_results.exc3_synchrony = zeros(n_conditions, num_repeats);
detailed_results.network_oscillation_power = zeros(n_conditions, num_repeats);
detailed_results.exc_inh_balance = zeros(n_conditions, num_repeats);

% *** New: Common input related data storage ***
detailed_results.common_input_series = cell(n_conditions, num_repeats);
detailed_results.exc3_time_series = cell(n_conditions, num_repeats);
detailed_results.fb_inh_time_series = cell(n_conditions, num_repeats);
detailed_results.exc3_spikes = cell(n_conditions, num_repeats);
detailed_results.signal_variance = zeros(n_conditions, num_repeats);
detailed_results.noise_variance = zeros(n_conditions, num_repeats);
detailed_results.effective_common_input = zeros(n_conditions, num_repeats);

% Detailed spike time data (for subsequent analysis)
detailed_results.spike_data = cell(n_conditions, num_repeats);

fprintf('Starting VIP disinhibition gradient experiment...\n');

for cond_idx = 1:n_conditions
    vip_activation = vip_activation_levels(cond_idx);
    fprintf('VIP activation strength: %.1f\n', vip_activation);
    
    for repeat = 1:num_repeats
        if mod(repeat, 10) == 0
            fprintf('  Repetition %d/%d\n', repeat, num_repeats);
        end
        
        % Setup neuron parameters
        [a, b, c, d, v, u] = setup_neuron_parameters(net, size(S_raw, 1));
        
        % Create experimental condition
        condition = struct(...
            'name', sprintf('VIP_Activation_%.1f', vip_activation), ...
            'input_layer1', 5, ...
            'input_layer2', 4, ...
            'input_layer3', 4, ...
            'vip_activation', vip_activation, ...
            'fb_excitability', 1.0, ...  % Fixed FB-Inh excitability
            'ff_strength', 1.0, ...
            'fb_strength', 1.0);
        
        % Run simulation (modified version, including common input recording)
        [firings, rates, spike_times, analysis_data] = ...
            run_vip_disinhibition_simulation_enhanced(v, u, a, b, c, d, S_raw, net, condition);
        
        % Collect basic activity data
        detailed_results.exc3_rates(cond_idx, repeat) = mean(rates(net.exc3));
        detailed_results.vip_rates(cond_idx, repeat) = mean(rates(net.vip));
        detailed_results.fb_inh_rates(cond_idx, repeat) = mean(rates(net.inh3_fb));
        detailed_results.ff_inh_rates(cond_idx, repeat) = mean(rates(net.inh3_ff));
        
        % *** New: Collect common input data ***
        detailed_results.common_input_series{cond_idx, repeat} = analysis_data.common_input_series;
        detailed_results.exc3_time_series{cond_idx, repeat} = analysis_data.exc3_time_series;
        detailed_results.fb_inh_time_series{cond_idx, repeat} = analysis_data.fb_inh_time_series;
        detailed_results.exc3_spikes{cond_idx, repeat} = analysis_data.exc3_spike_times;
        
        % Calculate signal-noise decomposition
        [sig_var, noise_var] = calculate_signal_noise_variance_vip(analysis_data.exc3_time_series);
        detailed_results.signal_variance(cond_idx, repeat) = sig_var;
        detailed_results.noise_variance(cond_idx, repeat) = noise_var;
        
        % Calculate effective common input
        detailed_results.effective_common_input(cond_idx, repeat) = ...
            calculate_effective_common_input_single_vip(analysis_data);
        
        vip_efficacy = calculate_feedback_efficacy_detailed(rates, net);
        detailed_results.vip_efficacy(cond_idx, repeat) = vip_efficacy;
        
        % Collect noise correlation data
        detailed_results.noise_correlations(cond_idx, repeat) = analysis_data.noise_correlation;
        detailed_results.pairwise_correlations(cond_idx, repeat) = analysis_data.pairwise_correlation;
        detailed_results.correlation_distribution_width(cond_idx, repeat) = analysis_data.correlation_width;
        
        % Collect VIP mechanism metrics
        detailed_results.vip_to_fb_efficacy(cond_idx, repeat) = analysis_data.vip_to_fb_efficacy;
        detailed_results.fb_suppression_index(cond_idx, repeat) = analysis_data.fb_suppression_index;
        detailed_results.disinhibition_index(cond_idx, repeat) = analysis_data.disinhibition_index;
        detailed_results.exc3_variability(cond_idx, repeat) = analysis_data.exc3_variability;
        detailed_results.condition = condition;
        
        % Collect network dynamics metrics
        detailed_results.exc3_synchrony(cond_idx, repeat) = analysis_data.exc3_synchrony;
        detailed_results.network_oscillation_power(cond_idx, repeat) = analysis_data.oscillation_power;
        detailed_results.exc_inh_balance(cond_idx, repeat) = analysis_data.exc_inh_balance;
        
        % Save spike data (sampled to save memory)
        if mod(repeat, 5) == 1  % Save detailed data every 5 times
            detailed_results.spike_data{cond_idx, repeat} = analysis_data.spike_times;
        end
    end
end

fprintf('VIP disinhibition gradient experiment completed.\n');
end

function [firings, rates, spike_times, analysis_data] = ...
    run_vip_disinhibition_simulation_enhanced(v, u, a, b, c, d, S_raw, net, condition)
% Core function for VIP disinhibition simulation (enhanced version, including common input recording)

T = 20000;  % 20 second simulation
dt = 0.5;
N = length(v);
firings = [];
spike_times = cell(N, 1);

% Dynamic variables
adaptation = zeros(N, 1);
tau_adapt = 200;
adaptation_strength = 0.03;
noise_scale = 0.4;

% Generate noise
noise_streams = randn(N, T);

% Record analysis data
analysis_data = struct();
exc3_spike_times = cell(length(net.exc3), 1);
vip_spike_times = cell(length(net.vip), 1);
fb_inh_spike_times = cell(length(net.inh3_fb), 1);

% Time series recording
exc3_time_series = zeros(1, floor(T/100));
vip_time_series = zeros(1, floor(T/100));
fb_inh_time_series = zeros(1, floor(T/100));
common_input_series = zeros(1, floor(T/100));  % New: Common input series
time_counter = 0;

fprintf('  Running VIP disinhibition simulation (VIP activation=%.1f)...\n', condition.vip_activation);

for t = 1:T
    % Calculate input (including VIP activation and common input)
    [I, common_input] = calculate_vip_disinhibition_input_with_common(t, N, noise_streams(:,t), noise_scale, net, condition);
    
    % Find firing neurons
    fired = find(v >= 30);
    if ~isempty(fired)
        firings = [firings; t+0*fired, fired];
        
        % Record spike times
        for i = 1:length(fired)
            spike_times{fired(i)} = [spike_times{fired(i)}; t*dt];
        end
        
        % Record spike times for specific populations
        exc3_fired = fired(ismember(fired, net.exc3));
        for neuron_id = exc3_fired'
            exc3_idx = find(net.exc3 == neuron_id, 1);
            if ~isempty(exc3_idx)
                exc3_spike_times{exc3_idx} = [exc3_spike_times{exc3_idx}; t*dt];
            end
        end
        
        vip_fired = fired(ismember(fired, net.vip));
        for neuron_id = vip_fired'
            vip_idx = find(net.vip == neuron_id, 1);
            if ~isempty(vip_idx)
                vip_spike_times{vip_idx} = [vip_spike_times{vip_idx}; t*dt];
            end
        end
        
        fb_fired = fired(ismember(fired, net.inh3_fb));
        for neuron_id = fb_fired'
            fb_idx = find(net.inh3_fb == neuron_id, 1);
            if ~isempty(fb_idx)
                fb_inh_spike_times{fb_idx} = [fb_inh_spike_times{fb_idx}; t*dt];
            end
        end
        
        % Reset firing neurons
        v(fired) = c(fired);
        u(fired) = u(fired) + d(fired);
        adaptation(fired) = adaptation(fired) + adaptation_strength;
    end
    
    % Synaptic input
    if t > 1 && ~isempty(fired)
        I_syn = sum(S_raw(:, fired), 2);
        I = I + I_syn;
    end
    
    % Update membrane potential
    I = I - adaptation;
    v = v + 0.5 * (0.04 * v.^2 + 5 * v + 140 - u + I);
    v = v + 0.5 * (0.04 * v.^2 + 5 * v + 140 - u + I);
    u = u + a.*(b.*v - u);
    
    % Update adaptation
    adaptation = adaptation + dt * (-adaptation / tau_adapt);
    
    % Record time series (every 100ms)
    if mod(t, 100) == 0
        time_counter = time_counter + 1;
        if time_counter <= length(exc3_time_series)
            recent_spikes = firings(firings(:,1) > t-100 & firings(:,1) <= t, :);
            exc3_time_series(time_counter) = sum(ismember(recent_spikes(:,2), net.exc3)) / 0.1;
            vip_time_series(time_counter) = sum(ismember(recent_spikes(:,2), net.vip)) / 0.1;
            fb_inh_time_series(time_counter) = sum(ismember(recent_spikes(:,2), net.inh3_fb)) / 0.1;
            common_input_series(time_counter) = common_input;  % Record common input
        end
    end
end

% Calculate firing rates
rates = zeros(N, 1);
for i = 1:N
    rates(i) = sum(firings(:,2) == i) / (T/1000);
end

% === Detailed analysis calculations ===

% 1. Noise correlation analysis
analysis_data.noise_correlation = calculate_improved_noise_correlation(exc3_spike_times, T*dt);
[pairwise_corr, corr_width] = calculate_detailed_pairwise_correlation(exc3_spike_times, T*dt);
analysis_data.pairwise_correlation = pairwise_corr;
analysis_data.correlation_width = corr_width;

% 2. VIP mechanism efficiency analysis
analysis_data.vip_to_fb_efficacy = calculate_vip_to_fb_efficacy(vip_spike_times, fb_inh_spike_times, rates, net);
analysis_data.fb_suppression_index = calculate_fb_suppression_index(rates, net, condition.vip_activation);
analysis_data.disinhibition_index = calculate_disinhibition_index(rates, net);

% 3. Network dynamics analysis
analysis_data.exc3_variability = calculate_exc3_variability(exc3_time_series);
analysis_data.exc3_synchrony = calculate_exc3_synchrony(exc3_spike_times);
analysis_data.oscillation_power = calculate_network_oscillation_power(exc3_time_series);
analysis_data.exc_inh_balance = calculate_exc_inh_balance(rates, net);

% 4. Save time series and spike data
analysis_data.exc3_time_series = exc3_time_series(exc3_time_series > 0);
analysis_data.vip_time_series = vip_time_series(vip_time_series > 0);
analysis_data.fb_inh_time_series = fb_inh_time_series(fb_inh_time_series > 0);
analysis_data.common_input_series = common_input_series(1:time_counter);  % New
analysis_data.exc3_spike_times = exc3_spike_times;  % New
analysis_data.spike_times = spike_times;
analysis_data.condition = condition;

fprintf('    Analysis completed: Noise correlation=%.4f, VIP efficacy=%.3f\n', ...
    analysis_data.noise_correlation, analysis_data.vip_to_fb_efficacy);
end

function [I, common_input] = calculate_vip_disinhibition_input_with_common(t, N, noise, noise_scale, net, condition)
% Calculate input including VIP disinhibition and return common input strength

% Base input
I = 1 + 0.2 * randn(N, 1);

% *** New: Common input component calculation ***
common_input_base = 2.0 * sin(2*pi*t/1000) + 1.0 * sin(2*pi*t/1500);
common_input = common_input_base * (1 + 0.3 * randn());  % Add some randomness

% Layer inputs
thalamic_input = condition.input_layer1 * (1 + 0.3 * sin(2*pi*t/1000));
I(net.exc1) = I(net.exc1) + thalamic_input;

if condition.input_layer3 > 0
    % *** Modified: Add common input component on top of original ***
    I(net.exc3) = I(net.exc3) + condition.input_layer3 * 0.6 + common_input * 0.15;  % Add appropriate common input
end

% === Key modification: Enhance FB-Inh baseline activity ===
% 1. Greatly enhance FB-Inh baseline excitatory input
fb_baseline_drive = 4.0;  % New: Strong baseline drive
I(net.inh3_fb) = I(net.inh3_fb) + fb_baseline_drive;

% 2. FB-Inh receives stronger input from Exc2 and Exc3 (simulating feedback connections)
exc_drive_to_fb = 1.5;  % Excitatory neuron drive to FB-Inh
I(net.inh3_fb) = I(net.inh3_fb) + exc_drive_to_fb * (1 + 0.2*sin(2*pi*t/1500));

% 3. Add FB-Inh specific background input
fb_background_input = 2.0 + 0.8 * randn(length(net.inh3_fb), 1);
I(net.inh3_fb) = I(net.inh3_fb) + fb_background_input;

% === Key: VIP activation ===
if isfield(condition, 'vip_activation') && condition.vip_activation > 0
    vip_drive = condition.vip_activation;
    
    % VIP neurons receive enhanced excitatory input
    base_vip_input = vip_drive * (1.2 + 0.5 * rand(length(net.vip), 1));
    I(net.vip) = I(net.vip) + base_vip_input;
    
    % Temporally modulated VIP input
    temporal_modulation = 1 + 0.3 * sin(2*pi*t/3000);
    I(net.vip) = I(net.vip) * temporal_modulation;
    
    % VIP-specific noise
    attention_noise = 0.5 * vip_drive * randn(length(net.vip), 1);
    I(net.vip) = I(net.vip) + attention_noise;
    
    % === Key: VIP inhibition effect on FB-Inh ===
    % When VIP is activated, reduce FB-Inh input (simulating inhibitory synapses)
    vip_inhibition_strength = vip_drive * 1.8;  % VIP inhibition strength
    
    % Apply VIP inhibition to each FB-Inh neuron
    for i = 1:length(net.inh3_fb)
        % Different FB-Inh have slightly different sensitivity to VIP
        sensitivity = 0.8 + 0.4 * (i-1)/max(1, length(net.inh3_fb)-1);  % 0.8-1.2
        vip_effect = vip_inhibition_strength * sensitivity;
        
        I(net.inh3_fb(i)) = I(net.inh3_fb(i)) - vip_effect;
    end
end

% FB-Inh baseline excitability (can be inhibited by VIP)
if isfield(condition, 'fb_excitability')
    fb_drive = condition.fb_excitability;
    I(net.inh3_fb) = I(net.inh3_fb) + fb_drive * 1.2;
end

% Keep other inhibitory neurons at low activity
I(net.inh1) = I(net.inh1) - 0.3;
I(net.inh2) = I(net.inh2) - 0.3;
I([net.inh3_ff, net.inh3_fb]) = I([net.inh3_ff, net.inh3_fb]) - 0.2;

% Add noise
I = I + noise * noise_scale;
I = max(I, 0);
end

function [sig_var, noise_var] = calculate_signal_noise_variance_vip(time_series)
% Calculate signal and noise variance (VIP version)

if length(time_series) < 10
    sig_var = NaN;
    noise_var = NaN;
    return;
end

% Low-pass filtering to extract signal component (<2Hz)
dt = 0.1; % 100ms sampling
fs = 1/dt; % 10Hz sampling frequency

% Simple moving average as low-pass filter
window_size = round(fs/2); % 0.5 second window
signal_component = movmean(time_series, window_size);

sig_var = var(signal_component);
noise_var = var(time_series - signal_component);
end

function effective_input = calculate_effective_common_input_single_vip(analysis_data)
% Calculate effective common input for single trial (VIP version)

if isfield(analysis_data, 'common_input_series') && ~isempty(analysis_data.common_input_series)
    % Effective common input = ratio of common input standard deviation to mean
    common_input_std = std(analysis_data.common_input_series);
    common_input_mean = mean(analysis_data.common_input_series);
    
    if common_input_mean ~= 0
        effective_input = common_input_std / abs(common_input_mean);
    else
        effective_input = common_input_std;
    end
else
    effective_input = NaN;
end
end

function improved_common_input_analysis = calculate_true_common_input_strength_variable_length_vip(detailed_results)
% Process common input strength calculation for variable-length time series under VIP conditions

[n_conditions, n_trials] = size(detailed_results.exc3_time_series);
improved_common_input_analysis = struct();

fprintf('\n=== Processing Variable-Length Time Series Data under VIP Conditions ===\n');

for cond = 1:n_conditions
    fprintf('VIP condition %d/%d...\n', cond, n_conditions);
    
    trial_activities = {};  % Use cell array to store variable-length sequences
    trial_lengths = [];
    
    % Collect time series from all trials
    for trial = 1:n_trials
        if ~isempty(detailed_results.exc3_time_series{cond, trial})
            exc3_data = detailed_results.exc3_time_series{cond, trial};
            if ~isempty(exc3_data)
                if size(exc3_data, 1) == 1
                    activity = exc3_data(1, :);  % [1 × length_i]
                elseif size(exc3_data, 2) == 1
                    activity = exc3_data(:, 1)';  % Transpose to row vector
                else
                    activity = mean(exc3_data, 1);  % For multiple neurons, take average
                end
                
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
fprintf('\n=== Data Length Statistics under VIP Conditions ===\n');
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

function [pairwise_corr, corr_width] = calculate_detailed_pairwise_correlation(spike_times, total_time)
% Detailed calculation of pairwise correlation and its distribution width

n_neurons = length(spike_times);
if n_neurons < 2
    pairwise_corr = NaN;
    corr_width = NaN;
    return;
end

% Use multiple time windows
window_sizes = [100, 200];  % 100ms and 200ms windows
all_correlations = [];

for window_size = window_sizes
    step_size = window_size * 0.5;
    n_windows = floor((total_time - window_size) / step_size) + 1;
    
    if n_windows < 10
        continue;
    end
    
    spike_counts = zeros(n_neurons, n_windows);
    
    % Calculate spike counts
    for win = 1:n_windows
        win_start = (win-1) * step_size;
        win_end = win_start + window_size;
        
        for neuron = 1:n_neurons
            if ~isempty(spike_times{neuron})
                spikes_in_window = spike_times{neuron}(...
                    spike_times{neuron} >= win_start & ...
                    spike_times{neuron} < win_end);
                spike_counts(neuron, win) = length(spikes_in_window);
            end
        end
    end
    
    % Filter active neurons
    mean_activity = mean(spike_counts, 2);
    std_activity = std(spike_counts, 0, 2);
    active_neurons = (mean_activity > 0.3) & (std_activity > 0.05);
    
    if sum(active_neurons) >= 2
        active_counts = spike_counts(active_neurons, :);
        
        % Standardization
        for i = 1:size(active_counts, 1)
            if std(active_counts(i, :)) > 0
                active_counts(i, :) = (active_counts(i, :) - mean(active_counts(i, :))) / std(active_counts(i, :));
            end
        end
        
        % Calculate correlation
        try
            corr_matrix = corrcoef(active_counts');
            n_active = size(corr_matrix, 1);
            upper_indices = find(triu(ones(n_active, n_active), 1));
            correlations = corr_matrix(upper_indices);
            valid_corr = correlations(~isnan(correlations) & abs(correlations) < 0.99);
            
            if ~isempty(valid_corr)
                all_correlations = [all_correlations; valid_corr];
            end
        catch
            % Skip when calculation fails
        end
    end
end

if ~isempty(all_correlations)
    pairwise_corr = mean(all_correlations);
    corr_width = std(all_correlations);  % Width of correlation distribution
else
    pairwise_corr = NaN;
    corr_width = NaN;
end
end

function efficacy = calculate_vip_to_fb_efficacy(vip_spike_times, fb_inh_spike_times, rates, net)
% Calculate VIP inhibition efficacy on FB-Inh

vip_activity = mean(rates(net.vip));
fb_inh_activity = mean(rates(net.inh3_fb));

if vip_activity > 0
    % Inverse relationship between VIP activity strength and FB-Inh activity
    efficacy = vip_activity / (1 + fb_inh_activity);
else
    efficacy = 0;
end

% If detailed spike time data is available, can calculate more precise causal relationship
% Using simplified version here
end

function suppression_index = calculate_fb_suppression_index(rates, net, vip_activation)
% Calculate degree of FB-Inh suppression

baseline_fb_rate = 8;  % Assumed baseline firing rate of FB-Inh
current_fb_rate = mean(rates(net.inh3_fb));

% Suppression index: degree of suppression relative to baseline
if baseline_fb_rate > 0
    suppression_index = (baseline_fb_rate - current_fb_rate) / baseline_fb_rate;
    suppression_index = max(0, suppression_index);  % Ensure non-negative
else
    suppression_index = 0;
end
end

function disinhibition_index = calculate_disinhibition_index(rates, net)
% Calculate disinhibition effect strength

exc3_activity = mean(rates(net.exc3));
fb_inh_activity = mean(rates(net.inh3_fb));
vip_activity = mean(rates(net.vip));

% Disinhibition index: VIP activity increases, FB-Inh activity decreases, Exc3 activity increases
if fb_inh_activity > 0 && vip_activity > 0
    disinhibition_index = (exc3_activity * vip_activity) / (1 + fb_inh_activity);
else
    disinhibition_index = 0;
end
end

function variability = calculate_exc3_variability(exc3_time_series)
% Calculate Exc3 activity variability

if length(exc3_time_series) > 1 && mean(exc3_time_series) > 0
    variability = std(exc3_time_series) / mean(exc3_time_series);
else
    variability = NaN;
end
end

function synchrony = calculate_exc3_synchrony(exc3_spike_times)
% Calculate Exc3 population synchrony

n_neurons = length(exc3_spike_times);
if n_neurons < 2
    synchrony = NaN;
    return;
end

% Simplified synchrony calculation: based on spike time variance
all_spikes = [];
for i = 1:n_neurons
    if ~isempty(exc3_spike_times{i})
        all_spikes = [all_spikes; exc3_spike_times{i}];
    end
end

if length(all_spikes) < 10
    synchrony = NaN;
    return;
end

all_spikes = sort(all_spikes);

% Calculate variability of adjacent spike intervals (inverse indicator of synchrony)
intervals = diff(all_spikes);
if length(intervals) > 1
    cv_intervals = std(intervals) / mean(intervals);
    synchrony = 1 / (1 + cv_intervals);  % Convert to synchrony metric
else
    synchrony = NaN;
end
end

function power = calculate_network_oscillation_power(time_series)
% Calculate network oscillation power

if length(time_series) < 20
    power = NaN;
    return;
end

% Simple oscillation power estimate: frequency domain analysis of time series
dt = 0.1;  % 100ms sampling
fs = 1/dt;  % Sampling frequency

% Detrend
detrended = detrend(time_series);

% FFT
Y = fft(detrended);
P = abs(Y).^2;
freqs = (0:length(Y)-1) * fs / length(Y);

% Focus on 0.5-30Hz frequency range (main range of neural oscillations)
freq_range = freqs >= 0.5 & freqs <= 30;
power = mean(P(freq_range));
end

function balance = calculate_exc_inh_balance(rates, net)
% Calculate excitation-inhibition balance

exc_activity = mean(rates([net.exc1, net.exc2, net.exc3]));
inh_activity = mean(rates([net.inh1, net.inh2, net.inh3_ff, net.inh3_fb]));

if exc_activity > 0
    balance = inh_activity / exc_activity;
else
    balance = NaN;
end
end

function analyze_vip_disinhibition_results(detailed_results, vip_activation_levels, net)
% Analyze VIP disinhibition experiment results

fprintf('\n=== VIP Disinhibition Regulation on Noise Correlation Analysis ===\n');

% Calculate means and standard deviations
mean_exc3 = mean(detailed_results.exc3_rates, 2);
std_exc3 = std(detailed_results.exc3_rates, 0, 2);
mean_vip = mean(detailed_results.vip_rates, 2);
std_vip = std(detailed_results.vip_rates, 0, 2);
mean_fb_inh = mean(detailed_results.fb_inh_rates, 2);
std_fb_inh = std(detailed_results.fb_inh_rates, 0, 2);

mean_noise_corr = mean(detailed_results.noise_correlations, 2, 'omitnan');
std_noise_corr = std(detailed_results.noise_correlations, 0, 2, 'omitnan');
mean_pairwise_corr = mean(detailed_results.pairwise_correlations, 2, 'omitnan');
mean_corr_width = mean(detailed_results.correlation_distribution_width, 2, 'omitnan');

mean_vip_efficacy = mean(detailed_results.vip_to_fb_efficacy, 2);
mean_fb_suppression = mean(detailed_results.fb_suppression_index, 2);
mean_disinhibition = mean(detailed_results.disinhibition_index, 2);

% Output results table
fprintf('\nBasic Activity Metrics:\n');
fprintf('VIP Act\tExc3 Firing Rate\t\tVIP Firing Rate\t\tFB-Inh Firing Rate\n');
fprintf('---------------------------------------------------------------\n');
for i = 1:length(vip_activation_levels)
    fprintf('%.1f\t\t%.2f±%.2f\t\t%.2f±%.2f\t\t%.2f±%.2f\n', ...
        vip_activation_levels(i), ...
        mean_exc3(i), std_exc3(i), ...
        mean_vip(i), std_vip(i), ...
        mean_fb_inh(i), std_fb_inh(i));
end

fprintf('\nNoise Correlation Metrics:\n');
fprintf('VIP Act\tNoise Correlation\t\tPairwise Correlation\t\tCorrelation Width\n');
fprintf('---------------------------------------------------------------\n');
for i = 1:length(vip_activation_levels)
    fprintf('%.1f\t\t%.4f±%.4f\t\t%.4f±%.4f\t\t%.4f\n', ...
        vip_activation_levels(i), ...
        mean_noise_corr(i), std_noise_corr(i), ...
        mean_pairwise_corr(i), mean(detailed_results.pairwise_correlations(i,:), 'omitnan'), ...
        mean_corr_width(i));
end

fprintf('\nVIP Mechanism Efficacy:\n');
fprintf('VIP Act\tVIP Efficacy\t\tFB Suppression Index\tDisinhibition Index\n');
fprintf('-------------------------------------------------------\n');
for i = 1:length(vip_activation_levels)
    fprintf('%.1f\t\t%.3f\t\t%.3f\t\t%.3f\n', ...
        vip_activation_levels(i), ...
        mean_vip_efficacy(i), mean_fb_suppression(i), mean_disinhibition(i));
end

% === Key findings analysis ===
fprintf('\n=== Key Findings ===\n');

% 1. Relationship between VIP activity and noise correlation
valid_idx = ~isnan(mean_noise_corr);
if sum(valid_idx) > 2
    [r_vip_noise, p_vip_noise] = corrcoef(vip_activation_levels(valid_idx)', mean_noise_corr(valid_idx));
    fprintf('1. VIP activation vs Noise correlation: r = %.3f, p = %.3f\n', r_vip_noise(1,2), p_vip_noise(1,2));
    
    if r_vip_noise(1,2) > 0.3 && p_vip_noise(1,2) < 0.05
        fprintf('   ✓ VIP activation significantly increases noise correlation (disinhibition effect)\n');
    elseif r_vip_noise(1,2) < -0.3 && p_vip_noise(1,2) < 0.05
        fprintf('   ✓ VIP activation significantly decreases noise correlation (unexpected finding)\n');
    else
        fprintf('   ? VIP effect on noise correlation is not significant\n');
    end
end

% 2. VIP inhibition effect on FB-Inh
[r_vip_fb, p_vip_fb] = corrcoef(vip_activation_levels', mean_fb_inh);
fprintf('2. VIP activation vs FB-Inh activity: r = %.3f, p = %.3f\n', r_vip_fb(1,2), p_vip_fb(1,2));

if r_vip_fb(1,2) < -0.3 && p_vip_fb(1,2) < 0.05
    fprintf('   ✓ VIP effectively suppresses FB-Inh activity\n');
else
    fprintf('   ✗ VIP suppression effect on FB-Inh is limited\n');
end

% 3. Relationship between FB-Inh and noise correlation
if sum(valid_idx) > 2
    [r_fb_noise, p_fb_noise] = corrcoef(mean_fb_inh(valid_idx), mean_noise_corr(valid_idx));
    fprintf('3. FB-Inh activity vs Noise correlation: r = %.3f, p = %.3f\n', r_fb_noise(1,2), p_fb_noise(1,2));
end

% 4. Disinhibition index analysis
max_disinhibition_idx = find(mean_disinhibition == max(mean_disinhibition), 1);
fprintf('4. Strongest disinhibition effect at VIP activation = %.1f\n', vip_activation_levels(max_disinhibition_idx));

% 5. Linear trend analysis
if length(vip_activation_levels) > 2 && sum(valid_idx) > 2
    p_noise = polyfit(vip_activation_levels(valid_idx), mean_noise_corr(valid_idx)', 1);
    fprintf('5. Noise correlation slope: %.4f /unit VIP activation\n', p_noise(1));
    
    p_exc3 = polyfit(vip_activation_levels, mean_exc3', 1);
    fprintf('6. Exc3 activity slope: %.3f Hz/unit VIP activation\n', p_exc3(1));
end
end

function create_vip_disinhibition_comprehensive_figure(detailed_results, vip_activation_levels, net)
% Create comprehensive VIP disinhibition analysis figure - add error bars

figure('Position', [50, 50, 1800, 1200]);

% Calculate means and standard deviations
mean_exc3 = mean(detailed_results.exc3_rates, 2);
std_exc3 = std(detailed_results.exc3_rates, 0, 2);
mean_vip = mean(detailed_results.vip_rates, 2);
std_vip = std(detailed_results.vip_rates, 0, 2);
mean_fb_inh = mean(detailed_results.fb_inh_rates, 2);
std_fb_inh = std(detailed_results.fb_inh_rates, 0, 2);

mean_noise_corr = mean(detailed_results.noise_correlations, 2, 'omitnan');
std_noise_corr = std(detailed_results.noise_correlations, 0, 2, 'omitnan');
mean_pairwise_corr = mean(detailed_results.pairwise_correlations, 2, 'omitnan');
std_pairwise_corr = std(detailed_results.pairwise_correlations, 0, 2, 'omitnan');
mean_vip_efficacy = mean(detailed_results.vip_to_fb_efficacy, 2);
std_vip_efficacy = std(detailed_results.vip_to_fb_efficacy, 0, 2);
mean_disinhibition = mean(detailed_results.disinhibition_index, 2);
std_disinhibition = std(detailed_results.disinhibition_index, 0, 2);

% 1. Main result: VIP activation vs Noise correlation
subplot(3, 4, 1);
errorbar(vip_activation_levels, mean_noise_corr, std_noise_corr, 'r-o', ...
    'LineWidth', 3, 'MarkerSize', 10, 'MarkerFaceColor', 'red');
xlabel('VIP Activation Strength');
ylabel('Noise Correlation');
title('A. Effect of VIP Activation on Noise Correlation');
grid on;

% Add trend line
valid_idx = ~isnan(mean_noise_corr);
if sum(valid_idx) > 2
    p = polyfit(vip_activation_levels(valid_idx), mean_noise_corr(valid_idx)', 1);
    x_fit = linspace(min(vip_activation_levels), max(vip_activation_levels), 100);
    hold on;
    plot(x_fit, polyval(p, x_fit), 'r--', 'LineWidth', 2);
    
    [r, p_val] = corrcoef(vip_activation_levels(valid_idx)', mean_noise_corr(valid_idx));
    text(0.1, 0.9, sprintf('r = %.3f\np = %.3f', r(1,2), p_val(1,2)), ...
         'Units', 'normalized', 'BackgroundColor', 'white', 'FontSize', 11, 'FontWeight', 'bold');
end

% 2. VIP activity level
subplot(3, 4, 2);
errorbar(vip_activation_levels, mean_vip, std_vip, 'm-s', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('VIP Activation Strength');
ylabel('VIP Firing Rate (Hz)');
title('B. VIP Neuron Activity Response');
grid on;

% 3. FB-Inh activity change
subplot(3, 4, 3);
errorbar(vip_activation_levels, mean_fb_inh, std_fb_inh, 'b-^', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('VIP Activation Strength');
ylabel('FB-Inh Firing Rate (Hz)');
title('C. FB-Inh Activity Suppression');
grid on;

% Add trend line
[r_vip_fb, p_vip_fb] = corrcoef(vip_activation_levels', mean_fb_inh);
if p_vip_fb(1,2) < 0.1
    p = polyfit(vip_activation_levels, mean_fb_inh', 1);
    x_fit = linspace(min(vip_activation_levels), max(vip_activation_levels), 100);
    hold on;
    plot(x_fit, polyval(p, x_fit), 'b--', 'LineWidth', 2);
    text(0.1, 0.9, sprintf('Slope = %.2f', p(1)), 'Units', 'normalized', ...
         'BackgroundColor', 'white', 'FontSize', 10);
end

% 4. Exc3 activity change
subplot(3, 4, 4);
errorbar(vip_activation_levels, mean_exc3, std_exc3, 'g-d', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('VIP Activation Strength');
ylabel('Exc3 Firing Rate (Hz)');
title('D. Exc3 Activity Disinhibition');
grid on;

% 5. VIP vs FB-Inh scatter plot (keep as is, scatter plots don't need errorbars)
subplot(3, 4, 5);
scatter(mean_vip, mean_fb_inh, 100, vip_activation_levels, 'filled', 'MarkerEdgeColor', 'black');
colorbar;
colormap(jet);
xlabel('VIP Firing Rate (Hz)');
ylabel('FB-Inh Firing Rate (Hz)');
title('E. VIP vs FB-Inh Relationship');

% Add fit line
[r, p_val] = corrcoef(mean_vip, mean_fb_inh);
if length(mean_vip) > 2
    p = polyfit(mean_vip, mean_fb_inh, 1);
    x_fit = linspace(min(mean_vip), max(mean_vip), 100);
    hold on;
    plot(x_fit, polyval(p, x_fit), 'k-', 'LineWidth', 2);
    text(0.1, 0.9, sprintf('r = %.3f\np = %.3f', r(1,2), p_val(1,2)), ...
         'Units', 'normalized', 'BackgroundColor', 'white', 'FontSize', 10);
end
grid on;

% 6. FB-Inh vs Noise correlation (keep as is, scatter plot)
subplot(3, 4, 6);
valid_idx = ~isnan(mean_noise_corr);
scatter(mean_fb_inh(valid_idx), mean_noise_corr(valid_idx), 100, vip_activation_levels(valid_idx), 'filled');
colorbar;
xlabel('FB-Inh Firing Rate (Hz)');
ylabel('Noise Correlation');
title('F. FB-Inh vs Noise Correlation');

if sum(valid_idx) > 2
    p = polyfit(mean_fb_inh(valid_idx), mean_noise_corr(valid_idx), 1);
    x_fit = linspace(min(mean_fb_inh(valid_idx)), max(mean_fb_inh(valid_idx)), 100);
    hold on;
    plot(x_fit, polyval(p, x_fit), 'k-', 'LineWidth', 2);
    
    [r, p_val] = corrcoef(mean_fb_inh(valid_idx), mean_noise_corr(valid_idx));
    text(0.1, 0.9, sprintf('r = %.3f\np = %.3f', r(1,2), p_val(1,2)), ...
         'Units', 'normalized', 'BackgroundColor', 'white', 'FontSize', 10);
end
grid on;

% 7. VIP mechanism efficacy
subplot(3, 4, 7);
errorbar(vip_activation_levels, mean_vip_efficacy, std_vip_efficacy, 'c-p', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('VIP Activation Strength');
ylabel('VIP→FB-Inh Efficacy');
title('G. VIP Mechanism Efficacy');
grid on;

% 8. Disinhibition index
subplot(3, 4, 8);
errorbar(vip_activation_levels, mean_disinhibition, std_disinhibition, 'k-h', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('VIP Activation Strength');
ylabel('Disinhibition Index');
title('H. Disinhibition Effect Strength');
grid on;

% 9. Pairwise correlation comparison
subplot(3, 4, 9);
valid_idx = ~isnan(mean_pairwise_corr);
if sum(valid_idx) > 0
    errorbar(vip_activation_levels(valid_idx), mean_pairwise_corr(valid_idx), std_pairwise_corr(valid_idx), 'y-*', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('VIP Activation Strength');
    ylabel('Pairwise Spike Correlation');
    title('I. Pairwise Neuron Correlation');
    grid on;
end

% 10. Network state temporal evolution (select representative conditions)
subplot(3, 4, 10);
sample_conditions = [1, ceil(length(vip_activation_levels)/2), length(vip_activation_levels)];
colors = ['b', 'g', 'r'];
legends = {};

for i = 1:length(sample_conditions)
    cond_idx = sample_conditions(i);
    % Calculate mean and standard deviation for all repeats in this condition
    condition_rates = detailed_results.exc3_rates(cond_idx, :);
    condition_rates = condition_rates(~isnan(condition_rates));
    
    if ~isempty(condition_rates)
        y_mean = mean(condition_rates);
        y_std = std(condition_rates);
        
        % Plot area of mean±standard deviation
        x_data = i;
        errorbar(x_data, y_mean, y_std, [colors(i) 'o'], 'LineWidth', 2, 'MarkerSize', 8);
        hold on;
        legends{end+1} = sprintf('VIP=%.1f (%.1f±%.1f)', vip_activation_levels(cond_idx), y_mean, y_std);
    end
end

xlabel('Condition');
ylabel('Exc3 Activity Strength (Hz)');
title('J. Exc3 Activity under Different VIP Activations');
if ~isempty(legends)
    legend(legends, 'Location', 'best');
end
grid on;

% 11. Multi-metric radar chart comparison (keep as is)
subplot(3, 4, 11);
% Select several key conditions for radar chart comparison
sample_indices = [1, ceil(length(vip_activation_levels)/3), ceil(2*length(vip_activation_levels)/3), length(vip_activation_levels)];
metrics = [mean_noise_corr(sample_indices)', mean_vip_efficacy(sample_indices)', ...
          mean_disinhibition(sample_indices)', mean_exc3(sample_indices)'/max(mean_exc3)];

% Simplified bar chart instead of radar chart
bar(metrics);
xlabel('VIP Activation Condition');
ylabel('Normalized Metric Value');
title('K. Multi-metric Comprehensive Comparison');
legend({'Noise Correlation', 'VIP Efficacy', 'Disinhibition Index', 'Normalized Exc3 Activity'}, 'Location', 'best');
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('%.1f', vip_activation_levels(x)), sample_indices, 'UniformOutput', false));
grid on;

% 12. Key findings summary (keep as is)
subplot(3, 4, 12);
axis off;

% Calculate key statistics
valid_idx = ~isnan(mean_noise_corr);
if sum(valid_idx) > 2
    [r_main, p_main] = corrcoef(vip_activation_levels(valid_idx)', mean_noise_corr(valid_idx));
    slope_noise = polyfit(vip_activation_levels(valid_idx), mean_noise_corr(valid_idx)', 1);
    
    % Display key findings
    text(0.1, 0.9, 'Key Findings:', 'FontSize', 14, 'FontWeight', 'bold');
    
    if r_main(1,2) > 0.3 && p_main(1,2) < 0.05
        result_text = '✓ VIP activation increases noise correlation';
        text(0.1, 0.8, result_text, 'FontSize', 12, 'Color', 'green');
    elseif r_main(1,2) < -0.3 && p_main(1,2) < 0.05
        result_text = '✓ VIP activation decreases noise correlation';
        text(0.1, 0.8, result_text, 'FontSize', 12, 'Color', 'blue');
    else
        result_text = '? VIP effect not significant';
        text(0.1, 0.8, result_text, 'FontSize', 12, 'Color', 'red');
    end
    
    text(0.1, 0.7, sprintf('Correlation coefficient: r = %.3f', r_main(1,2)), 'FontSize', 11);
    text(0.1, 0.6, sprintf('Significance: p = %.3f', p_main(1,2)), 'FontSize', 11);
    text(0.1, 0.5, sprintf('Rate of change: %.4f/unit', slope_noise(1)), 'FontSize', 11);
    
    % VIP mechanism efficacy
    max_efficacy = max(mean_vip_efficacy);
    text(0.1, 0.4, sprintf('Max VIP efficacy: %.3f±%.3f', max_efficacy, max(std_vip_efficacy)), 'FontSize', 11);
    
    % Disinhibition strength
    max_disinhibition = max(mean_disinhibition);
    text(0.1, 0.3, sprintf('Max disinhibition: %.3f±%.3f', max_disinhibition, max(std_disinhibition)), 'FontSize', 11);
    
    % Mechanism explanation
    text(0.1, 0.2, 'Mechanism:', 'FontSize', 12, 'FontWeight', 'bold');
    text(0.1, 0.1, 'VIP → Suppress FB-Inh → Disinhibit Exc3', 'FontSize', 10);
    text(0.1, 0.05, '→ Regulate noise correlation', 'FontSize', 10);
end

sgtitle('VIP Disinhibition Regulation Mechanism Analysis on Exc3 Noise Correlation (with Error Bars)', 'FontSize', 16, 'FontWeight', 'bold');

% Save figure
saveas(gcf, 'vip_disinhibition_comprehensive_analysis_with_errorbars.png');
saveas(gcf, 'vip_disinhibition_comprehensive_analysis_with_errorbars.fig');

fprintf('\nVIP disinhibition comprehensive analysis figure (with error bars) saved\n');
end

function analyze_vip_mechanism_details(detailed_results, vip_activation_levels, net)
% Detailed analysis of VIP disinhibition mechanism details

fprintf('\n=== VIP Disinhibition Mechanism Detailed Analysis ===\n');

% Calculate mean values of key metrics
mean_exc3_sync = mean(detailed_results.exc3_synchrony, 2, 'omitnan');
mean_oscillation = mean(detailed_results.network_oscillation_power, 2, 'omitnan');
mean_ei_balance = mean(detailed_results.exc_inh_balance, 2, 'omitnan');
mean_exc3_var = mean(detailed_results.exc3_variability, 2, 'omitnan');

fprintf('\nNetwork Dynamics Metrics Analysis:\n');
fprintf('VIP Act\tExc3 Synchrony\t\tOscillation Power\t\tE/I Balance\t\tExc3 Variability\n');
fprintf('-------------------------------------------------------------------------\n');
for i = 1:length(vip_activation_levels)
    fprintf('%.1f\t\t%.4f\t\t\t%.2f\t\t\t%.3f\t\t%.4f\n', ...
        vip_activation_levels(i), ...
        mean_exc3_sync(i), mean_oscillation(i), ...
        mean_ei_balance(i), mean_exc3_var(i));
end

% Mechanism efficacy analysis
fprintf('\n=== VIP Disinhibition Mechanism Efficacy Analysis ===\n');

% 1. VIP activation efficacy
mean_vip_rates = mean(detailed_results.vip_rates, 2);
for i = 1:length(vip_activation_levels)
    if vip_activation_levels(i) > 0
        vip_efficiency = mean_vip_rates(i) / vip_activation_levels(i);
        fprintf('VIP activation%.1f: VIP firing efficacy = %.2f Hz/unit activation\n', ...
            vip_activation_levels(i), vip_efficiency);
    end
end

% 2. FB-Inh suppression efficacy analysis
mean_fb_suppression = mean(detailed_results.fb_suppression_index, 2);
fprintf('\nFB-Inh suppression efficacy:\n');
for i = 1:length(vip_activation_levels)
    fprintf('VIP activation%.1f: FB suppression index = %.3f\n', ...
        vip_activation_levels(i), mean_fb_suppression(i));
end

% 3. Disinhibition conduction chain analysis
fprintf('\n=== Disinhibition Conduction Chain Efficacy ===\n');
mean_vip = mean(detailed_results.vip_rates, 2);
mean_fb_inh = mean(detailed_results.fb_inh_rates, 2);
mean_exc3 = mean(detailed_results.exc3_rates, 2);

for i = 1:min(5, length(vip_activation_levels))  % Show first 5 conditions
    if i > 1
        vip_change = mean_vip(i) - mean_vip(1);
        fb_change = mean_fb_inh(i) - mean_fb_inh(1);
        exc3_change = mean_exc3(i) - mean_exc3(1);
        
        fprintf('VIP activation%.1f relative to baseline:\n', vip_activation_levels(i));
        fprintf('  VIP change: +%.2f Hz\n', vip_change);
        fprintf('  FB-Inh change: %.2f Hz\n', fb_change);
        fprintf('  Exc3 change: +%.2f Hz\n', exc3_change);
        
        if vip_change > 0
            fb_sensitivity = abs(fb_change) / vip_change;
            fprintf('  FB-Inh sensitivity to VIP: %.3f\n', fb_sensitivity);
        end
        
        if abs(fb_change) > 0
            exc3_sensitivity = exc3_change / abs(fb_change);
            fprintf('  Exc3 sensitivity to FB-Inh: %.3f\n', exc3_sensitivity);
        end
        fprintf('\n');
    end
end

% 4. Noise correlation regulation efficacy
mean_noise_corr = mean(detailed_results.noise_correlations, 2, 'omitnan');
valid_idx = ~isnan(mean_noise_corr);

if sum(valid_idx) > 2
    fprintf('=== Noise Correlation Regulation Efficacy ===\n');
    
    % Calculate range of noise correlation change
    noise_corr_range = max(mean_noise_corr(valid_idx)) - min(mean_noise_corr(valid_idx));
    vip_range = max(vip_activation_levels) - min(vip_activation_levels);
    
    fprintf('Noise correlation change range: %.4f\n', noise_corr_range);
    fprintf('VIP activation range: %.1f\n', vip_range);
    fprintf('Regulation efficacy: %.4f correlation change/unit VIP activation\n', noise_corr_range/vip_range);
    
    % Analyze regulation linearity
    [r, p] = corrcoef(vip_activation_levels(valid_idx)', mean_noise_corr(valid_idx));
    fprintf('Regulation linearity: r = %.3f, p = %.3f\n', r(1,2), p(1,2));
    
    if abs(r(1,2)) > 0.7
        fprintf('✓ VIP has strong regulatory capability on noise correlation\n');
    elseif abs(r(1,2)) > 0.4
        fprintf('? VIP has moderate regulatory capability on noise correlation\n');
    else
        fprintf('✗ VIP has limited regulatory capability on noise correlation\n');
    end
end

% 5. Optimal VIP activation level analysis
fprintf('\n=== Optimal VIP Activation Level Analysis ===\n');

% Find optimal points for different metrics
[~, max_disinhibition_idx] = max(mean(detailed_results.disinhibition_index, 2));
[~, max_vip_efficacy_idx] = max(mean(detailed_results.vip_to_fb_efficacy, 2));

fprintf('Maximum disinhibition effect: VIP activation = %.1f\n', vip_activation_levels(max_disinhibition_idx));
fprintf('Maximum VIP efficacy: VIP activation = %.1f\n', vip_activation_levels(max_vip_efficacy_idx));

if sum(valid_idx) > 2
    if r(1,2) > 0  % Positive correlation
        [~, max_noise_corr_idx] = max(mean_noise_corr(valid_idx));
        fprintf('Maximum noise correlation: VIP activation = %.1f\n', vip_activation_levels(max_noise_corr_idx));
    else  % Negative correlation
        [~, min_noise_corr_idx] = min(mean_noise_corr(valid_idx));
        fprintf('Minimum noise correlation: VIP activation = %.1f\n', vip_activation_levels(min_noise_corr_idx));
    end
end
end

function create_vip_mechanism_figure(detailed_results, vip_activation_levels)
% Create VIP mechanism detailed analysis figure - add error bars

figure('Position', [100, 100, 1600, 1000]);

% Calculate mean values and standard deviations of metrics
mean_exc3_sync = mean(detailed_results.exc3_synchrony, 2, 'omitnan');
std_exc3_sync = std(detailed_results.exc3_synchrony, 0, 2, 'omitnan');

mean_oscillation = mean(detailed_results.network_oscillation_power, 2, 'omitnan');
std_oscillation = std(detailed_results.network_oscillation_power, 0, 2, 'omitnan');

mean_ei_balance = mean(detailed_results.exc_inh_balance, 2, 'omitnan');
std_ei_balance = std(detailed_results.exc_inh_balance, 0, 2, 'omitnan');

mean_exc3_var = mean(detailed_results.exc3_variability, 2, 'omitnan');
std_exc3_var = std(detailed_results.exc3_variability, 0, 2, 'omitnan');

mean_vip_efficacy = mean(detailed_results.vip_to_fb_efficacy, 2);
std_vip_efficacy = std(detailed_results.vip_to_fb_efficacy, 0, 2);

mean_fb_suppression = mean(detailed_results.fb_suppression_index, 2);
std_fb_suppression = std(detailed_results.fb_suppression_index, 0, 2);

% 1. Exc3 synchrony change
subplot(2, 4, 1);
valid_idx = ~isnan(mean_exc3_sync);
if sum(valid_idx) > 0
    errorbar(vip_activation_levels(valid_idx), mean_exc3_sync(valid_idx), std_exc3_sync(valid_idx), 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('VIP Activation Strength');
    ylabel('Exc3 Synchrony');
    title('A. Exc3 Population Synchrony');
    grid on;
end

% 2. Network oscillation power
subplot(2, 4, 2);
valid_idx = ~isnan(mean_oscillation);
if sum(valid_idx) > 0
    errorbar(vip_activation_levels(valid_idx), mean_oscillation(valid_idx), std_oscillation(valid_idx), 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('VIP Activation Strength');
    ylabel('Oscillation Power');
    title('B. Network Oscillation Strength');
    grid on;
end

% 3. E/I balance
subplot(2, 4, 3);
valid_idx = ~isnan(mean_ei_balance);
if sum(valid_idx) > 0
    errorbar(vip_activation_levels(valid_idx), mean_ei_balance(valid_idx), std_ei_balance(valid_idx), 'g-^', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('VIP Activation Strength');
    ylabel('E/I Ratio');
    title('C. Excitation-Inhibition Balance');
    grid on;
end

% 4. Exc3 activity variability
subplot(2, 4, 4);
valid_idx = ~isnan(mean_exc3_var);
if sum(valid_idx) > 0
    errorbar(vip_activation_levels(valid_idx), mean_exc3_var(valid_idx), std_exc3_var(valid_idx), 'm-d', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('VIP Activation Strength');
    ylabel('Exc3 Variability (CV)');
    title('D. Exc3 Activity Stability');
    grid on;
end

% 5. VIP efficacy temporal evolution
subplot(2, 4, 5);
errorbar(vip_activation_levels, mean_vip_efficacy, std_vip_efficacy, 'c-p', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('VIP Activation Strength');
ylabel('VIP→FB-Inh Efficacy');
title('E. VIP Mechanism Efficacy');
grid on;

% 6. FB suppression strength
subplot(2, 4, 6);
errorbar(vip_activation_levels, mean_fb_suppression, std_fb_suppression, 'k-h', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('VIP Activation Strength');
ylabel('FB Suppression Index');
title('F. FB-Inh Suppression Degree');
grid on;

% 7. Mechanism efficacy heatmap
subplot(2, 4, 7);
mechanism_matrix = [mean_vip_efficacy'; mean_fb_suppression'; 
                   mean(detailed_results.disinhibition_index, 2)'];
imagesc(mechanism_matrix);
colormap(hot);
colorbar;
set(gca, 'YTickLabel', {'VIP Efficacy', 'FB Suppression', 'Disinhibition'});
set(gca, 'XTick', 1:2:length(vip_activation_levels));
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('%.1f', x), vip_activation_levels(1:2:end), 'UniformOutput', false));
xlabel('VIP Activation Strength');
title('G. Mechanism Efficacy Heatmap');

% 8. Mechanism schematic diagram (keep as is)
subplot(2, 4, 8);
axis off;

% Draw simplified neural network schematic
% VIP neuron
rectangle('Position', [0.1, 0.7, 0.2, 0.2], 'FaceColor', 'm', 'EdgeColor', 'k');
text(0.2, 0.8, 'VIP', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% FB-Inh neuron
rectangle('Position', [0.4, 0.7, 0.2, 0.2], 'FaceColor', 'b', 'EdgeColor', 'k');
text(0.5, 0.8, 'FB-Inh', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% Exc3 neuron
rectangle('Position', [0.7, 0.7, 0.2, 0.2], 'FaceColor', 'r', 'EdgeColor', 'k');
text(0.8, 0.8, 'Exc3', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% Connection arrows
% VIP -> FB-Inh (inhibition)
annotation('arrow', [0.32, 0.38], [0.8, 0.8], 'Color', 'red', 'LineWidth', 2);
text(0.35, 0.85, 'Inhibition', 'HorizontalAlignment', 'center', 'Color', 'red');

% FB-Inh -> Exc3 (inhibition)
annotation('arrow', [0.62, 0.68], [0.8, 0.8], 'Color', 'blue', 'LineWidth', 2);
text(0.65, 0.85, 'Inhibition', 'HorizontalAlignment', 'center', 'Color', 'blue');

% Exc3 self-connection (noise correlation)
annotation('arrow', [0.8, 0.85], [0.75, 0.7], 'Color', 'green', 'LineWidth', 2);
annotation('arrow', [0.85, 0.8], [0.7, 0.75], 'Color', 'green', 'LineWidth', 2);
text(0.8, 0.6, 'Noise Correlation', 'HorizontalAlignment', 'center', 'Color', 'green');

% Title and description
text(0.5, 0.5, 'VIP Disinhibition Mechanism', 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
text(0.5, 0.4, '1. VIP activation', 'HorizontalAlignment', 'center', 'FontSize', 10);
text(0.5, 0.35, '2. Suppress FB-Inh', 'HorizontalAlignment', 'center', 'FontSize', 10);
text(0.5, 0.3, '3. Disinhibit Exc3', 'HorizontalAlignment', 'center', 'FontSize', 10);
text(0.5, 0.25, '4. Regulate noise correlation', 'HorizontalAlignment', 'center', 'FontSize', 10);

xlim([0, 1]);
ylim([0, 1]);
title('H. VIP Disinhibition Mechanism Schematic');

sgtitle('Detailed Dynamic Analysis of VIP Disinhibition Mechanism (with Error Bars)', 'FontSize', 16, 'FontWeight', 'bold');

% Save figure
saveas(gcf, 'vip_mechanism_detailed_analysis_with_errorbars.png');
saveas(gcf, 'vip_mechanism_detailed_analysis_with_errorbars.fig');

fprintf('\nVIP mechanism detailed analysis figure (with error bars) saved\n');
end

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
prob_fb_inhibition = 0.8;    % Reduced from 0.8 to 0.5 (critical!)
prob_vip_disinhibition = 0.5; % Reduced from 0.5 to 0.4

% *** Key modification 2: Adjust weight ranges ***
exc_weight = @() max(0, 6 + 2*randn());    
inh_weight = @() -max(0, 8 + 3*randn());   
% FB-Inh weight range narrowed and added E/I balance-based regulation
fb_inh_weight = @(scale_factor) -max(0, (6 + 2*randn()) * scale_factor);  % Dynamic adjustment
vip_weight = @() -max(0, 9 + 3*randn());   

% ========================
% Feedforward connection construction (unchanged)
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

% *** Key: Regulate FB-Inh weights based on excitatory input strength ***
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
                    weight_variation = 0.7 + 0.6*rand(); 
                    S_raw(exc_neuron, j) = -individual_weight * weight_variation;
                end
            end
        end
    end
end

% *** Modification 4: Reduce lateral inhibition among FB-Inh ***
for i = 1:length(inh3_fb)
    for j = 1:length(inh3_fb)
        if i ~= j && rand() <= 0.4  
            S_raw(inh3_fb(i), inh3_fb(j)) = -max(0, 4 + 1*randn()); 
        end
    end
end

% *** Modification 5: Simplify delayed inhibition pathway ***
for i = 1:length(exc3)
    for j = 1:length(inh3_fb)
        if rand() <= 0.1 
            S_raw(inh3_fb(j), exc3(i)) = exc_weight() * 0.4; 
        end
    end
end

% ========================
% VIP circuit
% ========================
for i = vip
    for j = [exc2, exc3]
        if rand() <= prob_between * 0.6  
            S_raw(i, j) = exc_weight() * 0.7;  
        end
    end
end

for i = inh3_fb
    for j = vip
        if rand() <= prob_vip_disinhibition
            S_raw(i, j) = vip_weight()*0.3;  
        end
    end
end

for i = inh3_ff
    for j = vip
        if rand() <= prob_vip_disinhibition * 0.2  
            S_raw(i, j) = vip_weight() * 0;  
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
% *** Network balance check and adjustment ***
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

function efficacy = calculate_feedback_efficacy_detailed(rates, net)
% Detailed calculation of feedback inhibition efficacy

% Method 1: FB-Inh inhibition efficacy on Exc3
fb_inh_activity = mean(rates(net.inh3_fb));
exc3_activity = mean(rates(net.exc3));

if exc3_activity > 0
    efficacy = fb_inh_activity / exc3_activity;
else
    efficacy = 0;
end
end