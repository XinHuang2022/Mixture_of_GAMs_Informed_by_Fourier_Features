%% Subroutine for conjugate gradient method 
    function y = applyRegularizedMatrix( A, lambda, x )
	    % Compute (A^T A + lambda I) * x
	    y = A' * ( A * x ) + lambda * x;
    end