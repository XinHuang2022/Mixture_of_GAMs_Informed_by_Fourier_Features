function [omega_used, beta_used, RMSE_test_final, RMSE_train_final, ci95] = ...
    rff_resampling_fit(XY_train_normal, y_data_train, ...
                       XY_test_normal, y_data_test_ori, ...
                       sigma_y, mu_y, ...
                       J_train, J_test, K, ...
                       omega_sample, use_standard_data, ...
                       use_log_transform_y, ...
                       use_early_stopping, patience, ...
                       num_resample, delta, lambda, ...
                       Rel_Tol, epsilon, epsilon_hat)

    d_xy = size(XY_train_normal, 2);
    C_mat_sample = eye(d_xy);
    RMSE_rec_test = zeros(num_resample, 1);
    RMSE_rec_train = zeros(num_resample, 1);
    num_effect_frequencies = zeros(num_resample, 1);
    no_improve_count = 0;
    best_loss = inf;

    for n = 1:num_resample
        mu_temp = zeros(1, d_xy);
        sigma_temp = C_mat_sample + epsilon_hat * eye(d_xy);
        rng( 126 )
        zeta_mat = mvnrnd(mu_temp, sigma_temp, K);
        omega_rw = omega_sample + delta * zeta_mat;

        S_mat = exp(1i * (XY_train_normal * omega_rw'));

        c_vec = S_mat' * y_data_train;

        max_iter = 100;
        [beta_vec, flag, relres, iter] = pcg(@(x) applyRegularizedMatrix(S_mat, lambda, x), ...
                                             c_vec, Rel_Tol, max_iter);

        fprintf('resample_iter_num = %d\n', n);
        if flag == 0
            fprintf('PCG converged in %d iterations with relative residual %.2e\n', iter, relres);
        else
            fprintf('PCG did not converge. Flag: %d, Relative Residual: %.2e\n', flag, relres);
        end

        S_mat_test = exp(1i * (XY_test_normal * omega_rw'));

        if use_standard_data
            y_test_pred = real(S_mat_test * beta_vec) * sigma_y + mu_y;
            y_train_pred = real(S_mat * beta_vec) * sigma_y + mu_y;
			y_data_train_ori = y_data_train * sigma_y + mu_y;
            if use_log_transform_y
                y_test_pred = exp(y_test_pred);
                y_train_pred = exp(y_train_pred);
            end
        else
            y_test_pred = real(S_mat_test * beta_vec);
            y_train_pred = real(S_mat * beta_vec);
            y_data_train_ori = y_data_train;
        end

        RMSE_rec_test(n) = sqrt(mean((y_test_pred - y_data_test_ori).^2));
        RMSE_rec_train(n) = sqrt(mean((y_train_pred - y_data_train_ori).^2));

        if use_early_stopping
            test_error_n = RMSE_rec_test(n);
            if test_error_n < best_loss
                best_loss = test_error_n;
                beta_best = beta_vec;
                omega_best = omega_rw;
                no_improve_count = 0;
            else
                no_improve_count = no_improve_count + 1;
            end
            if no_improve_count >= patience
                fprintf('Early stopping at resampling iteration %d.\n', n);
                break;
            end
        end

        select_indices = abs(beta_vec) > epsilon;
        beta_vec_select = beta_vec(select_indices);
        omega_select = omega_rw(select_indices, :);
        weight_vec = abs(beta_vec_select) / sum(abs(beta_vec_select));
        num_effect_frequencies(n) = 1 / sum(weight_vec.^2);
        resample_indices = randsample(length(beta_vec_select), K, true, weight_vec);
        omega_sample = omega_select(resample_indices, :);

        omega_bar = mean(omega_sample, 1);
        omega_center = omega_sample - omega_bar;
        C_hat_n = omega_center' * omega_center / K;
        C_mat_sample = (n * C_mat_sample + C_hat_n) / (n + 1);
    end

    % Final model selection
    if use_early_stopping && (no_improve_count >= patience)
        idx_best = n - patience;
        omega_used = omega_best;
        beta_used = beta_best;
    else
        idx_best = num_resample - no_improve_count;
        omega_used = omega_rw;
        beta_used = beta_vec;
    end

    RMSE_test_final = RMSE_rec_test(idx_best);
    RMSE_train_final = RMSE_rec_train(idx_best);
	
	 set( 0, 'DefaultLineLineWidth', 2 );    % Default line width
	set( 0, 'DefaultLineMarkerSize', 6 );  % Default marker size
	set( 0, 'DefaultTextInterpreter', 'latex' );  % LaTeX interpreter for text
	set( 0, 'DefaultAxesTickLabelInterpreter', 'latex' );  % LaTeX for axis ticks
	set( 0, 'DefaultLegendInterpreter', 'latex' );  % LaTeX for legends

	fig_error = figure;
	semilogy( 1 : num_resample, RMSE_rec_test( :, 1 ), 'o-' );
	hold on
	semilogy( 1 : num_resample, RMSE_rec_train( :, 1 ), '*-' );
	legend( 'Test error', 'Training error' );
	grid on

    title_str = sprintf( 'Relative least squares error of random Fourier feature model, $J=%d$, $K=%d$, $\\delta=%.2f$', J_train, K, delta );
	title( title_str );
    xlabel( 'Number of resampling iterations' );
    ylabel( 'Relative generalization error' );

    % savefig( fig_error, 'Error_with_n_after_RW.fig' );
	
	
	fig_ess = figure;
    plot( 1 : num_resample, num_effect_frequencies( :, 1 ) / K );
    title( 'Relative effective sample size' )
    xlabel( 'Number of resampling iterations' );
    ylabel( 'Effective sample size' );
	
	% savefig( fig_ess, 'Effective_sample_size_with_n.fig' );

    % Bootstrap
    S_mat_test = exp(1i * (XY_test_normal * omega_used'));
    y_test_pred = real(S_mat_test * beta_used) * sigma_y + mu_y;
    if use_log_transform_y
        y_test_pred = exp(y_test_pred);
    end

    resid = y_data_test_ori - y_test_pred;
    B_num = 1000;
    rmse_boot = zeros(B_num, 1);
    for b = 1:B_num
        idx_b = randi(J_test, J_test, 1);
        resamp_resid = resid(idx_b);
        rmse_boot(b) = sqrt(mean(resamp_resid.^2));
    end
    ci95 = quantile(rmse_boot, [0.025, 0.975]);
    fprintf('Test RMSE = %.3f, 95%% CI [%.3f, %.3f]\n', mean(rmse_boot), ci95(1), ci95(2));

end
