function varargout = irlba(varargin)
%
% IRLBA will find a few singular values and singular vectors of A 
% such that A*V = U*S, where V is an orthonormal n x k matrix of "right" singular 
% vectors, U is an orthonormal p x k matrix of "left" singular vectors, and S is 
% a k x k diagonal matrix of singular values. 
%
% [U,S,V,FLAG] = IRLBA(A,OPTIONS) 
% [U,S,V,FLAG] = IRLBA('Afunc',P,N,OPTIONS) 
%
% The first input argument into IRLBA can be a numeric matrix A or an M-file 
% ('Afunc'). If the P x N matrix A is a filename then Y = Afunc(X,P,N,'transpose'). 
% If transpose = 'F', then X is an (N x 1) matrix and Y = A*X. If transpose = 'T', 
% then X is an (P x 1) matrix and Y = A'*X.
% 
% OUTPUT OPTIONS:
% ---------------
%
% I.)   IRLBA(A) 
%       Displays the desired singular values.    
%
% II.)  S = IRLBA(A) 
%       Returns the desired singular values in the vector S. 
%
% III.) [U,S,V] = IRLBA(A)
%       S is a diagonal matrix that contains the desired singular values in descending
%       order along the diagonal, the matrix V contains the corresponding "right" singular
%       vectors, and U contains the corresponding "left" singular vectors such that 
%       that A*V = U*S, V'*V = I, and U'*U = I.  
%
% IV.)  [U,S,V,FLAG] = IRLBA(A)
%       This option returns the same as (III) plus a two dimensional array FLAG 
%       that reports if the algorithm converges and the number of matrix vector 
%       products. If FLAG(1) = 0 then this implies normal return: all desired 
%       singular values have converged. If FLAG(1) = 1 then the maximum number 
%       of iterations have been reached before all desired values converged. 
%       FLAG(2) contains the number of matrix vector products (A*X and A'*X) 
%       used by the code. If the maximum number of iterations are reached 
%       (FLAG(1) = 1), then the matrices U, V, and S contain any singular pairs
%       that have converged plus the last singular pair approximation for the 
%       pairs that have not converged.
%
% INPUT OPTIONS:
% --------------
%                                   
%       ... = IRLBA(A,OPTS)
%       OPTS is a structure containing input parameters. The input parameters can
%       be given in any order. The structure OPTS may contain some or all of the 
%       following input parameters. The string for the input parameters can contain
%       upper or lower case characters. 
%       
%  INPUT PARAMETER      DESCRIPTION    
%
%
%  OPTS.ADJUST       Initial number of vectors to add to the k restart vectors. After
%                    vectors start to converge more vectors are added to help 
%                    increase convergence.
%                    DEFAULT VALUE    OPTS.ADJUST = 3
%
%  OPTS.AUG          Vectors used to augment the Krylov space. Choices are
%                    either Ritz vectors or Harmonic Ritz vectors.
%                    DEFAULT VALUE    OPTS.AUG = 'HARM'  if SIGMA = 'SS'
%                                     OPTS.AUG = 'RITZ'  if SIGMA = 'LS'                 
%                        
%  OPTS.DISPS        Indicates if k approximate singular values and residuals are to be 
%                    displayed on each iteration. Set positive to display the values.
%                    DEFAULT VALUE   DISPS = 0 
%
%  OPTS.K            Number of desired singular values.             
%                    DEFAULT VALUE  K = 6
%
%  OPTS.MAXIT        Maximum number of iterations, i.e. maximum number of restarts.                           
%                    DEFAULT VALUE  MAXIT = 1000                      
%
%  OPTS.M            Size of the Lanczos bidiagonal matrix. The program may increase
%                    M to ensure certain requirements are satisfied. A warning message
%                    will be displayed if M increases.                          
%                    DEFAULT VALUE    M = 20
%
%  OPTS.REORTH       Three letter string specifying whether to use one-sided full reorthogonalization
%                    or two-sided. One-sided is performed only on the "short" vectors.
%                    Two-sided orthogonality is used when cond(A) estimated by cond(B) > 1/sqrt(eps).
%                    'ONE'  One-sided full reorthogonalization.
%                    'TWO'  Two-sided full reorthogonalization.
%                    DEFAULT VALUE  REORTH = 'ONE' 
%
%  OPTS.SIGMA        Two letter string specifying the location of the desired singular values.            
%                    'LS'  Largest singular values.
%                    'SS'  Smallest singular values.                 
%                    DEFAULT VALUE   SIGMA = 'LS'                                                         
%                                 
%  OPTS.TOL          Tolerance used for convergence. Convergence is determined when             
%                    || A*V - U*S || <= TOL*||A||. Where ||A|| is approximated by                                   
%                    the largest singular value of all projection matrices.  
%                    DEFAULT VALUE    TOL = 1d-6 
%                                                              
%  OPTS.V            Initial N x 1 starting vector for the Lanczos bidiagonal method,
%                    when P >= N and sigma = 'LS' or 'SS'. If P < N and sigma = 'SS' then
%                    V must be a P x 1 starting vector. This avoids computing zero 
%                    eigenvalues of A'A. 
%                    DEFAULT VALUE  V = []
% 
%  DATE:  11/08/06
%  VER:  1.0

%  AUTHORS:
%  James Baglama     University of Rhode Island  E-mail: jbaglama@math.uri.edu
%  Research supported by NSF grant DMS-0311786.
%
%  Lothar Reichel    Kent State University       E-mail: reichel@mcs.kent.edu

% REFERENCES:
% 1.) "Augmented Implicitly Restarted Lanczos Bidiagonalization Methods", 
%      J. Baglama and L. Reichel, SIAM J. Sci. Comput., 27 (2005), pp. 19-42.
% 2.) J. Baglama and L. Reichel, "Restarted Block Lanczos Bidiagonalization
%     Methods", Numerical Algorithms, 43 (2006) 251-272. 

% Values used in the GUI demo IRLBABLKDEMO. 
% Not needed for command line computation.
global guiresidual FLAG matrixprod dnbls err output waithan;
string1 = 'plot(0.5,0.5,''Color'',[0.8 0.8 0.8]);';
string2 = 'set(gca,''XTick'',[],''YTick'',[],''Visible'',''off'');';
string3 = 'text(0,0.5,err,''FontSize'',10,''Color'',''r'');';
string4 = 'axis([0 1 0 1]);';
output  = strcat(string1,string2,string3,string4);
guiresidual = [];

% Incorrect number of output arguments requested.
if (nargout >= 5 || nargout==2 ), error('ERROR: Incorrect number of output arguments.'); end

%----------------------------%
% BEGIN: PARSE INPUT VALUES. %
%----------------------------%

% No input arguments, return help.
if nargin == 0, help irlba, return, end

% Matrix A is stored in varargin{1}. Check type (numeric or character) and dimensions.
if (isstruct(varargin{1})) 
    err = 'A must be a matrix.';
    if ~isempty(gcbo), close(waithan); eval(output); end, error(err);
end
if ~isnumeric(varargin{1})
   if nargin == 1, error('Need dimension P for matrix A).'), end  
   p = varargin{2};
   if ~isnumeric(p) || length(p) ~= 1 
      error('Second argument P must be a numeric value.'); 
   end
   if nargin == 2, error('Need dimension N for matrix A).'); end  
   n = varargin{3}; 
   if ~isnumeric(n) || length(n) ~= 1 
      error('Third argument N must be a numeric value.'); 
   end
else
   [p,n] = size(varargin{1});
end

% Set all input options to default values.
adjust = 3; disps = 0;  k = 6; maxit = 1000; m = 20; sigma = 'LS';  tol = 1d-6; aug = [];
reorth='ONE'; V=[];

% Get input options from the structure array.
if nargin > 1 + 2*(~isnumeric(varargin{1}))
   options = varargin{2+2*(~isnumeric(varargin{1})):nargin};
   names = fieldnames(options); lnames = lower(names);
   [nrows,ncols] = size(names); 
   for i=1:nrows
       eval([lnames{i},'=','options.(names{',num2str(i),'});'])
   end 
end

% Check type of input values and output an error message if needed.
if (~isnumeric(adjust) || ~isnumeric(disps)   || ~isnumeric(k)  || ~isnumeric(maxit) || ...
    ~isnumeric(m)    || ~isnumeric(tol) || ~ischar(sigma) || ~ischar(reorth))
   error('Incorrect type for input value(s) in the structure.');
end

% Check the values of sigma.
if length(sigma)  ~= 2, error('SIGMA must be LS or SS'); end
if (~strcmp(sigma,'SS') && ~strcmp(sigma,'LS') )
   error('SIGMA must be LS or SS.'); 
end

% Check the values of reorth.
if length(reorth) ~=3, error('REORTH must be ONE or TWO.'); end
if (~strcmp(reorth,'ONE') && ~strcmp(reorth,'TWO') )
   error('REORTH must be ONE or TWO.'); 
end

% Interchange p and n so that size(A'A) = min(p,n). Avoids
% finding zero values when searching for the smallest singular
% values.
interchange = 0; if n > p && strcmp(sigma,'SS'), t=p; p=n; n=t; interchange = 1; end 

% Preallocate memory for W and F. These matrices are full and resizing will cause
% an increase in cpu time.
W = zeros(p,m); F = zeros(n,1);

% If starting matrix V0 is not given then set starting matrix V0 to be a 
% (n x 1) matrix of normally distributed random numbers.
if isempty(V)
  V = zeros(n,m); % Preallocate memory for V. This matrix is a full matrix.
  V(:,1) = randn(n,1); 
else
  V(:,2:m) = zeros(n,m-1);
end

% Increase the number of desired values by adjust to help increase convergence. k
% is re-adjusted as vectors converge. This is only an initial value of k.
K_org = k; k = k + adjust; 

% Check for input errors in the data structure.
if k       <= 0,   error('k must be a positive value.'),    end
if k  >  min(n,p), error('k must be less than min(n,p) + %g.',adjust),  end   
if m     < 1,   error('m must be greater than 0.'),   end
if tol     <  0,   error('TOL must be non-negative.'),      end
if maxit   <= 0,   error('MAXIT must be positive.'),        end
if m >= min(n,p)
   m = floor(min(n,p)-0.1);
   warning(['Changing M to ',num2str(m)]);
   if m - k - 1  < 0
      adjust = 0; k = m - 1; 
      warning(['Changing adjust to ',num2str(adjust)]);
      warning(['Changing k to ',num2str(k)]);
   end
end
if m - k - 1  < 0
   m = ceil(k+1+0.1); 
   warning(['Changing M to ',num2str(m)]);
end
if m >= min(n,p)
   m = floor(min(n,p)-0.1);
   adjust = 0; k = m - 1; 
   warning(['Changing adjust to ',num2str(adjust)]);
   warning(['Changing k to ',num2str(k)]);
   warning(['Changing M to ',num2str(m)]);
end
if ~isnumeric(V), error('Incorrect starting matrix V0.'), end
if (size(V,1) ~= n), error('Incorrect size of starting matrix V0.'), end
 
% Determine which vectors to use for augmenting.
if isempty(aug)
  if strcmp(sigma,'LS'), aug = 'RITZ'; else aug = 'HARM'; end
else
  if length(aug) ~= 4, error('Unknown value for AUG. AUG must be RITZ or HARM'); end
  if ~ischar(aug), error('Unknown value for AUG. AUG must be RITZ or HARM'); end
  if (~strcmp(aug,'RITZ') && ~strcmp(aug,'HARM') )
     error('Unknown value for AUG. AUG must be RITZ or HARM'); 
  end
end

% Set tolerance to machine precision if tol < eps.
if tol < eps, tol = eps; end

%--------------------------%
% END: PARSE INPUT VALUES. %
%--------------------------%

%-----------------------------------------------------------%
% BEGIN: DESCRIPTION AND INITIALIZATION OF LOCAL VARIABLES. %
%-----------------------------------------------------------%

% Initialization and description of local variables.   
B = [];              % Bidiagonal matrix.
Bsz =[];             % Size of the bidiagonal matrix.
conv = 'F';          % Boolean to determine if all desired singular values have converged.
eps23 = eps^(2/3);   % Two thirds of eps, used for Smax. Avoids using zero.
I=[];                % Used for indexing.
iter = 1;            % Main loop iteration count.
J=[];                % Used for indexing.
mprod = 0;           % The number of matrix vector products.
R_F =[];             % Two norm of the residual vector F.
sqrteps = sqrt(eps); % Square root of machine tolerance used in convergence testing.
Smax = 1;            % Holds the maximum value of all computed singular values of B est. ||A||_2.
Smin=[];             % Holds the minimum value of all computed singular values of B est. cond(A).
SVTol = min(sqrteps,tol);  % Tolerance to determine when a singular vector has converged.
S_B =[];             % Singular values of B.
U_B =[];             % Left singular vectors of B.
V_B =[];             % Right singular vectors of B.
V_B_last=[];         % Holds the last row of the modified V_B.
S_B2 =[];            % Singular values of [B ||F||].
U_B2 =[];            % Left singular vectors of [B ||F||].
V_B2 =[];            % Right singular vectors of [B ||F||].

% Value use for demo only, holds the change in m. 
% Not needed for command line computation.
dnbls = m;

%--------------------------------------------------------------------%
% END: DESCRIPTION AND INITIALIZATION OF LOCAL AND GLOBAL VARIABLES. %
%--------------------------------------------------------------------%

%----------------------------%
% BEGIN: MAIN ITERATION LOOP %
%----------------------------%

while (iter <= maxit)
    
    % Compute the Lanczos bidiagonalization decomposition.
    [V,W,F,B,mprod] = ablanzbd(varargin{1},V,W,F,B,k,interchange,m,n,p,mprod,SVTol*Smax,reorth,iter);

    % Determine the size of the bidiagonal matrix B.
    Bsz = size(B,1);  

    % Compute the norm of the vector F, and normalize F.
    R_F = norm(F); F = F/R_F; 

    % Compute singular triplets of B. MATLAB's svds orders the singular values
    % largest to smallest.
    [U_B,S_B,V_B] = svd(B); S_B = diag(S_B);
    
    % Estimate ||A|| using the largest singular value over all iterations
    % and estimate the cond(A) using approximations to the largest and smallest 
    % singular values. If a small singular value is less than sqrteps use only Ritz
    % vectors to augment and require two-sided reorthogonalization.
    if iter==1
       Smax = S_B(1); Smin = S_B(Bsz);
    else 
       Smax = max(Smax,S_B(1)); Smin = min(Smin,S_B(Bsz));
    end
    Smax = max(eps23,Smax);
    if Smin/Smax < sqrteps, reorth = 'TWO'; aug = 'RITZ'; end
   
    % Re-order the singular values accordingly. MATLAB's SVD orders the 
    % singular values largest to smallest.
    if strcmp(sigma,'SS'), I = Bsz:-1:1; U_B = U_B(:,I); V_B = V_B(:,I); S_B = S_B(I); end   
   
    % Compute the residuals for the singular values.
    R = R_F*U_B(Bsz,:);  
  
    % Convergence tests and displays.
    [conv,U_B,S_B,V_B,k] = convtests(Bsz,disps,tol,K_org,U_B,S_B,V_B,abs(R),iter,k,SVTol,Smax);

    % If all desired singular values converged then exit main loop.
    if strcmp(conv,'T'),  break,  end;

    % Reached maximum number of iterations, exit main loop.
    if iter >= maxit, break, end;
    
    % Compute the starting vectors and first block B(1:k,1:k+1). 
    if strcmp(aug,'HARM')

       % Update the SVD of B to be the SVD of [B ||F||E_m].
       [U_B2,S_B,V_B2] = svd([diag(S_B) R']); S_B = diag(S_B);   
       if strcmp(sigma,'SS')
          U_B2 = U_B2(:,I); V_B2 = V_B2(:,I); S_B = S_B(I);
       end
       U_B = U_B*U_B2; 
       V_B = [[V_B; zeros(1,Bsz)] flipud(eye(Bsz+1,1))]*V_B2;
       V_B_last = V_B(Bsz+1,1:k); % Set equal to the last row of V_B.
       s = R_F*(B\flipud(eye(Bsz,1))); 
       V_B = V_B(1:Bsz,:) + s*V_B(Bsz+1,:);
     
       % Vectors V_B are not orthogonal.
       [V_B,R] = qr([ [V_B(:,1:k); zeros(1,k)]  [-s; 1] ],0);
       V(:,1:k+1) = [V F]*V_B; 
  
       % Update and compute the k x k+1 part of B.
       B = diag(S_B(1:k))*triu((R(1:k+1,1:k)+ R(:,k+1)*V_B_last)');      
       %B = B(1:k,1:k+1)/R;  Change B = diag(S_B(1:k+1));  

    else
      V(:,1:k+1) = [V*V_B(:,1:k) F];    
      B = [diag(S_B(1:k)), R(1:k)'];
    end

    % Compute left approximate singular vectors.
    W(:,1:k) = W*U_B(:,1:k);

    % Update the main iteration loop count.
    iter = iter + 1;
    
    % Wait bar used in the GUI demo IRLBABLKDEMO. 
    % Not used for command line computation.
   if ~isempty(gcbo),  waitbar(iter/maxit), end 

end 

%--------------------------%
% END: MAIN ITERATION LOOP %
%--------------------------%

%-----------------------%
% BEGIN: OUTPUT RESULTS %
%-----------------------%

% Test to see if maximum number of iterations have been reached.
FLAG(1) = 0; if iter >= maxit, FLAG(1) = 1; end; 

% Output option I: Display singular values only.
if (nargout == 0)
  if FLAG(1)==1, disp('Maximum number of iterations exceeded.'); end
  SingularValues = S_B(1:K_org)
end

% Output option II: Set singular values equal to output vector.  
if (nargout == 1), varargout{1} = S_B(1:K_org); end

% Output option III and IV: Output singular triplets (U,S,V), where
% A*V = U*S.
if nargout > 1
   if interchange
      varargout{1} = V*V_B(:,1:K_org); 
      varargout{2} = diag(S_B(1:K_org)); 
      varargout{3} = W*U_B(:,1:K_org);
   else
      varargout{1} = W*U_B(:,1:K_org); 
      varargout{2} = diag(S_B(1:K_org)); 
      varargout{3} = V*V_B(:,1:K_org);
   end
   if nargout == 4, FLAG(2) = mprod; varargout{4} = FLAG; end
end

% Used in the GUI demo IRLBABLKDEMO. 
% Not used for command line computation.
% Outputs number of matrix-vector products.
matrixprod = mprod;

%---------------------%
% END: OUTPUT RESULTS %
%---------------------%

%------------------------------------------------%
% BEGIN: LANCZOS BIDIAGONALIZATION DECOMPOSITION %
%------------------------------------------------%

function [V,W,F,B,mprod] = ablanzbd(A,V,W,F,B,k,interchange,m,n,p,mprod,SVTol,reorth,iter)
% Computes the Lanczos bidiagonalization decomposition
%  
%  A*V  = W*B
%  A'*W = V*B' + F*E^T
%  
% with full reorthogonalization. The matrix A can be passed as a numeric 
% matrix or as a filename. If the P x N matrix A is a filename then
% Y = Afunc(X,P,N,'transpose'). If transpose = 'F', then X is an (N x 1)
% matrix and Y = A*X. If transpose = 'T', then X is an (P x 1) matrix
% and Y = A'*X.

% James Baglama
% DATE: 3/05/04

% Initialization of main loop count J, output matrix B, singular block indicator.
J = 1;  

% Normalize starting vector.
if iter == 1
   V(:,1) = V(:,1)/norm(V(:,1)); B=[];
else
   J = k+1;
end

% Matrix A product with vector(s) V, (W=A*V).
if interchange
   if ~isnumeric(A)
     W(:,J) = feval(A,V(:,J),p,n,'T');
  else
     W(:,J) = (V(:,J)'*A)';
  end
else
  if ~isnumeric(A)
     W(:,J) = feval(A,V(:,J),p,n,'F');
  else
     W(:,J) = A*V(:,J);
  end
end

% Count the number of matrix vector products.
mprod = mprod + 1;

% Input vectors are singular vectors and AV(:,J) which must be orthogonalized.
if iter ~= 1
   W(:,J) = orthog(W(:,J),W(:,1:J-1));
end

% Compute the norm of W.
S = norm(W(:,J)); 

% Check for linearly dependent vectors.
if S <= SVTol
   W(:,J) = randn(size(W,1),1);
   W(:,J) = orthog(W(:,J),W(:,1:J-1));
   W(:,J) = W(:,J)/norm(W(:,J));
   S = 0;
else
   W(:,J) = W(:,J)/S;
end

% Begin of main iteration loop for the block Lanczos bidiagonalization decomposition.
while (J <= m)

   % Matrix A' product with vector(s), (F = A'*W). 
   if interchange
      if ~isnumeric(A)
         F = feval(A,W(:,J),p,n,'F');
      else
         F = A*W(:,J);
      end
   else
      if ~isnumeric(A)
        F = feval(A,W(:,J),p,n,'T');
      else
        F = (W(:,J)'*A)';
      end
   end
     
   % Count the number of matrix vector products.
   mprod = mprod + 1;
   
   % One step of the block classical Gram-Schmidt process.
   F = F - V(:,J)*S';

   % Full Reorthogonalization step. "Short vectors".
   F = orthog(F,V(:,1:J));
   
   if J+1 <= m

      % Compute the norm of F.
      R = norm(F); 
      
      % Check for linearly dependent vectors.
      if R <= SVTol
         F = randn(size(V,1),1); 
         F = orthog(F,V(:,1:J));
         V(:,J+1) = F/norm(F);
         R = 0;
      else
        V(:,J+1) = F/R;
      end 

      % Compute block bidiagonal matrix B. 
      if isempty(B) 
         B = [S R'];
      else
         B = [B zeros(J-1,1); zeros(1,J-1) S R'];
      end 
    
      % Matrix A product with vector(s), (W=A*V).
      if interchange
         if ~isnumeric(A)
           W(:,J+1) = feval(A,V(:,J+1),p,n,'T');
         else
           W(:,J+1) = (V(:,J+1)'*A)';
         end
      else
         if ~isnumeric(A)
            W(:,J+1) = feval(A,V(:,J+1),p,n,'F');
         else
            W(:,J+1) = A*V(:,J+1);
         end
      end

      % Count the number of matrix vector products.
      mprod = mprod + 1;

      % One step of the block classical Gram-Schmidt process.
      W(:,J+1) =  W(:,J+1) - W(:,J)*R';

      % Full Reorthogonalization step. "Long vectors"
      if ( iter == 1 || strcmp(reorth,'TWO') )
         W(:,J+1) = orthog(W(:,J+1),W(:,1:J));
      end
   
      % Compute the norm of W.
      S = norm(W(:,J+1)); 
      
     % Check for linearly dependent vectors.
     if S <= SVTol 
        W(:,J+1) = randn(size(W,1),1);
        W(:,J+1) = orthog(W(:,J+1),W(:,1:J));
        W(:,J+1) = W(:,J+1)/norm(W(:,J+1));
        S = 0;
     else
        W(:,J+1) = W(:,J+1)/S;
     end 

   else
    
    % Add last block to matrix B
     B = [B; zeros(1,J-1) S]; 

   end

    % Update iteration count.
    J = J + 1;

end

function Y = orthog(Y,X)
% Orthogonalize vectors Y against vectors X.

if size(X,2) < size(Y,2), dotY = X'*Y; else dotY = (Y'*X)'; end
Y = Y - X*dotY; 

 
%----------------------------------------------------%
% END: BLOCK LANCZOS BIDIAGONALIZATION DECOMPOSITION %
%----------------------------------------------------%

%--------------------------%
% BEGIN: CONVERGENCE TESTS %
%--------------------------%

function [conv,U_B,S_B,V_B,k] = convtests(Bsz,disps,tol,K_org,U_B,S_B,V_B,residuals,iter,k,SVTol,Smax)
% This function checks the convergence of Singular values. 

% James Baglama
% DATE: 3/05/04

% Residual values used in the GUI demo IRLBABLKDEMO. 
% Not needed for command line computation.
global guiresidual; guiresidual(iter,:) = residuals(1:K_org);

% Set convergence to false.
conv = 'F';  

% Output intermediate results.
 if disps ~= 0 
   disp(sprintf('  Singular Value     Residual       Iteration: %d',iter));
   S = sprintf('%15.5e %15.5e \n',[S_B(1:K_org)';residuals(1:K_org)]);
   disp(S); disp(' '); disp(' ');
 end

% Check for convergence.
Len_res = length(find(residuals(1:K_org)' < tol*Smax));
if Len_res == K_org && iter > 1
   U_B = U_B(:,1:K_org); V_B = V_B(:,1:K_org); S_B = S_B(1:K_org);
   
   % Set convergence to true.
   conv = 'T'; return;
   
end

% Adjust k to include more vectors as the number of vectors converge.
Len_res = length(find(residuals(1:K_org)' < SVTol*Smax));
k = max(k, K_org + Len_res); if k > Bsz - 3, k = Bsz - 3; end

%------------------------%
% END: CONVERGENCE TESTS %
%------------------------%
