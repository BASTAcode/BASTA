!*******************************************************************************
!
!     SUBROUTINES TO FIT THE SIGNATURES OF ACOUSTIC GLITCHES IN THE SECOND
!     DIFFERENCES
!     --------------------------------------------------------------------------
!
!     AUTHOR NAME     : KULDEEP VERMA
!     EMAIL ADDRESSES : verma@ice.csic.es, kuldeep@phys.au.dk,
!                       kuldeepv89@gmail.com
!     LAST MODIFIED   : 16/12/2021
!
!*******************************************************************************
!
!     freqDif2 : (input) Input second differences - l, n, nu(muHz), err(muHz),
!                dif2(muHz), err(muHz)
!     icov : (input) Inverse covariance matrix for second differences
!     num_of_dif2 : (input) Number of second differences
!     acoustic_radius : (input) An estimate of acoustic radius (in s)
!     tauHe : (input) A guess for the acoustic depth of the helium ionization
!             zone (in s)
!     dtauHe : (input) Defines the range of search for tauHe,
!              range - (tauHe - dtauHe, tauHe + dtauHe)
!     tauCZ : (input) A guess for the acoustic depth of the convection zone
!             base (in s)
!     dtauCZ : (input) Defines the range of search for tauCZ,
!              range - (tauCZ - dtauCZ, tauCZ + dtauCZ)
!     npoly_sd : (optional) Degree of the polynomial used + 1
!     total_num_of_param_sd : (optional) Total number of fitting parameter
!     num_guess : (optional) Number of initial guesses used in the global
!                 minimum search
!     nderiv_sd : (optional) Derivative order used in the regularization
!     tol_grad_sd : (optional) Tolerance on the gradient of the cost function
!     regu_param_sd : (optional) Regularization parameter
!     param : (output) Array containing the fitted parameters
!     chi2 : (output) Chi-square of the fit
!     reg : (output) Regularization term
!     ier : (output) Error parameter, ier=0 implies successful execution
!           ier=-1 implies that routine failed to fit
!
!*******************************************************************************

      SUBROUTINE FIT_SD(freqDif2,icov,num_of_dif2,acoustic_radius,&
            tauHe,dtauHe,tauCZ,dtauCZ,npoly_sd,total_num_of_param_sd,&
            num_guess,nderiv_sd,tol_grad_sd,regu_param_sd,param,chi2,&
            reg,ier)
      IMPLICIT NONE

      REAL*8, PARAMETER :: MU = 1.d-6, PI = 3.141592653589793d0
      REAL*8, PARAMETER :: reps = 1.d-14, aeps = 0.d0

      INTEGER :: npoly_sd, total_num_of_param_sd
      INTEGER :: nderiv_sd, num_guess
      INTEGER :: i, i0, j, nfun, ier, num_of_dif2
      INTEGER :: iseed
      INTEGER, ALLOCATABLE :: seed(:)

      REAL*8 :: freqDif2(num_of_dif2,6)
      REAL*8 :: icov(num_of_dif2,num_of_dif2)
      REAL*8 :: acoustic_radius
      REAL*8 :: tauHe, tauCZ, dtauHe, dtauCZ
      REAL*8 :: chi2_total, chi2, reg, chi2_total_new, chi2_new, reg_new
      REAL*8 :: tol_grad_sd, regu_param_sd
      REAL*8 :: loga, logb
      REAL*8 :: param(total_num_of_param_sd)
      REAL*8 :: dparam(total_num_of_param_sd,2)
      REAL*8 :: par(total_num_of_param_sd)
      REAL*8 :: ran_num(total_num_of_param_sd)
      REAL*8 :: grad(total_num_of_param_sd)
      REAL*8 :: hess(total_num_of_param_sd,total_num_of_param_sd)
      REAL*8 :: scratch(3*total_num_of_param_sd)
!f2py intent(in)  :: freqDif2, icov, num_of_dif2
!f2py intent(in)  :: acoustic_radius, tauHe, dtauHe, tauCZ, dtauCZ
!f2py intent(out) :: param, chi2, reg, ier
!f2py INTEGER :: npoly_sd = 3, total_num_of_param_sd = 10
!f2py INTEGER :: nderiv_sd = 1, num_guess = 200
!f2py REAL*8  :: tol_grad_sd = 1.e-2, regu_param_sd = 1.e3


      ! Check for inconsistencies in the input parameters
      IF (total_num_of_param_sd .ne. npoly_sd + 7) THEN
        WRITE(*,'(A50)') 'ERROR: inconsistent argument for glitch fit!'
        STOP
      ENDIF

      ! Set the seed for the random number generator
      CALL RANDOM_SEED(size=iseed)
      ALLOCATE(seed(iseed))
      seed = 13579
      call RANDOM_SEED(put=seed)

      ! Estimate the parameter ranges to find the global minimum
      CALL SEARCH_SPACE_SD(dparam)

      chi2_total = 1.d99
      chi2 = 1.d99
      reg  = 0.d0
      i0 = total_num_of_param_sd - 7
      DO i = 1, num_guess

        ! Initial guess
        CALL RANDOM_NUMBER(ran_num)
        DO j = 1, total_num_of_param_sd
          IF (j .EQ. i0+1 .OR. j .EQ. i0+4) THEN
            loga = LOG10(dparam(j,1))
            logb = LOG10(dparam(j,2))
            par(j) = loga + (logb - loga) * ran_num(j)
            par(j) = 10.d0**par(j)
          ELSE
            par(j) = dparam(j,1) + (dparam(j,2) - dparam(j,1)) * ran_num(j)
          ENDIF
        ENDDO

        par(i0+1) = SQRT(par(i0+1))
        par(i0+4) = SQRT(par(i0+4))
        par(i0+5) = SQRT(8.d0 * PI**2 * par(i0+5)**2)
        CALL BFGS(total_num_of_param_sd,par,chi2_total_new,grad,hess,&
             nfun,reps,aeps,ier,FCN_SD,scratch)
        par(i0+1) = par(i0+1)**2
        par(i0+3) = MODULO(par(i0+3),2*PI)
        par(i0+4) = par(i0+4)**2
        par(i0+5) = SQRT(par(i0+5)**2/(8.d0 * PI**2))
        par(i0+7) = MODULO(par(i0+7),2*PI)

        ! Update the fit only if:
        !--------------------------
        ! (1) Deeper than the previous local minimum,
        !     i.e. chi2_total_new < chi2_total
        ! (2) Equally good or better fit than the previous one,
        !     i.e. chi2_new =< chi2
        ! (3) maximum value of derivatives w.r.t. parameters is < 1d-2,
        ! (4) Acoustic width of He ionization zone > 0.1s,
        ! (5) acoustic depths of CZ and He > 0 s,
        ! (6) acoustic depth of CZ > acoustic depth of He,
        ! (7) acoustic depth of CZ < acoustic radius.
        reg_new = REGU_SD(par)
        chi2_new = chi2_total_new - reg_new
        IF (chi2_total_new .LT. chi2_total .AND. &
            chi2_new .LE. chi2 .AND. &
            MAXVAL(ABS(grad(:))) .LT. tol_grad_sd .AND. &
            par(i0+5) .GT. 0.1d0 .AND. &
            par(i0+2) .GT. 0.d0 .AND. par(i0+6) .GT. 0.d0 .AND. &
            par(i0+2) .GT. par(i0+6) .AND. &
            par(i0+2) .LT. acoustic_radius) THEN
          chi2_total = chi2_total_new
          chi2  = chi2_new
          reg   = reg_new
          param = par
        ENDIF

      ENDDO

      !Check if the data was fitted
      ier = 0
      IF (chi2 .GT. 1.d98) THEN
        ier = -1
        chi2 = -1.d0
        reg = -1.d0
        param(:) = (dparam(:, 1) + dparam(:, 2)) / 2.d0
      ENDIF
      DEALLOCATE(seed)



      CONTAINS



      !*************************************************************************
      !TO INITIALIZE THE PARAMETER SPACE TO SEARCH FOR GLOBAL MINIMUM
      !*************************************************************************
      !
      !dparam : (output) Array containing the parameter space to look for the
      !         global minimum
      !
      !*************************************************************************

      SUBROUTINE SEARCH_SPACE_SD(dparam)
      IMPLICIT NONE

      INTEGER :: ns

      REAL*8 :: dparam(total_num_of_param_sd,2)


      ! parameters associated with smooth component
      dparam = 0.d0
      dparam(1,1) = -1.d0
      dparam(1,2) = 1.d0
      dparam(2,1) = -1.d-3
      dparam(2,2) = 1.d-3
      dparam(3,1) = -1.d-6
      dparam(3,2) = 1.d-6

      ! parameters associated with CZ signature
      ns = total_num_of_param_sd - 7
      dparam(ns+1,1) = 1.d4
      dparam(ns+1,2) = 1.d7
      dparam(ns+2,1) = MAX(0.d0, tauCZ - dtauCZ)
      dparam(ns+2,2) = MIN(acoustic_radius, tauCZ + dtauCZ)
      dparam(ns+3,1) = 0.d0
      dparam(ns+3,2) = 6.28d0

      ! parameters associated with He signature
      dparam(ns+4,1) = 1d-5
      dparam(ns+4,2) = 1.d-1
      dparam(ns+5,1) = 20.d0
      dparam(ns+5,2) = 160.d0
      dparam(ns+6,1) = MAX(0.d0, tauHe - dtauHe)
      dparam(ns+6,2) = MIN(acoustic_radius, tauHe + dtauHe)
      dparam(ns+7,1) = 0.d0
      dparam(ns+7,2) = 6.28d0

      END SUBROUTINE SEARCH_SPACE_SD



      !*************************************************************************
      !TO MINIMISE A FUNCTION OF SEVERAL VARIABLES USING QUASI-NEWTON METHOD.
      !*************************************************************************
      !
      !N : (input) Number of variables
      !X : (input/output) Real array of length N containing the initial
      !	guess for the minimum.
      !	After execution it should contain the coordinates of minimiser
      !F : (output) The function value at X
      !G : (output) Real array of length N containing the gradient vector at X
      !H : (output) Real array of length N*N containing the estimated
      !	Hessian matrix at X. The first dimension of H should be N.
      !NUM : (output) Number of function evaluations used by the subroutine
      !REPS : (input) Required relative accuracy
      !AEPS : (input) Required absolute accuracy, all components of the
      !	Minimiser will be calculated with accuracy MAX(AEPS, REPS*ABS(X(I)))
      !IER : (output) Error parameter, IER=0 implies successful execution
      !	IER=53 implies that Hessian matrix at final point is probably singular
      !		The iteration may have converged to a saddle point
      !	IER=503 implies that N < 1, in which case no calculations are done
      !	IER=526 implies that iteration failed to converge to specified accuracy
      !	Other values may be set by LINMIN
      !FCN : (input) Name of the subroutine to calculate the function value
      !     and its derivatives
      !WK : Real array of length 3N used as scratch space
      !
      !SUBROUTINE FCN(N,X,F,G) to calculate the required function, must be supplied
      !	by the user. Here N is the number of variables, F is the
      !	function value at X and G is the gradient vector. X and G
      !	are real arrays of length N. F and G must be calculated by FCN.
      !
      !Required routines : LINMIN, FLNM, FCN
      !
      !*************************************************************************

      SUBROUTINE BFGS(N,X,F,G,H,NUM,REPS,AEPS,IER,FCN,WK)
      IMPLICIT NONE

      LOGICAL :: QC

      INTEGER, PARAMETER :: NIT=200
      INTEGER :: N, NUM, IER, IER1
      INTEGER :: I, J, IT, N2

      REAL*8 :: F, F1, DF, DF1, REPS, AEPS
      REAL*8 :: DG, GHG, GI, H1, H2, R1, X1, X2
      REAL*8 :: X(N), WK(3*N), H(N,N), G(N)

      EXTERNAL :: FCN


      IER=0
      IF(N.LT.1) THEN
        IER=503
        RETURN
      ENDIF

      DO 2000 I=1,N
      !Initialise the Hessian matrix to unit matrix
        DO 1800 J=1,N
1800    H(J,I)=0.0
        H(I,I)=1.0
2000  CONTINUE
      !If some variable needs to be kept fixed the corresponding
      !diagonal elements, H(I,I) should be set to zero.

      CALL FCN(N,X,F,G)
      DF=ABS(F)
      IF(DF.EQ.0.0) DF=1
      N2=2*N
      NUM=1
      H2=1.

      !The iteration loop
      DO 4000 IT=1,NIT
        DF1=0.0
        H1=H2
        H2=0.0
      !Calculating the search direction WK =S^(k)
        DO 2400 I=1,N
          WK(I)=0.0
          DO 2200 J=1,N
            H2=H2+ABS(H(I,J))
2200      WK(I)=WK(I)-H(I,J)*G(J)
          DF1=DF1+WK(I)*G(I)
2400    CONTINUE

        IF(DF1.EQ.0.0) THEN
      !If gradient vanishes, then quit
      !If Hessian matrix appears to be singular, set the error flag
          IF(ABS(H2/H1).GT.1.3D0) IER=53
          RETURN
        ENDIF

      !Initial guess for line search
        X1=0
        X2=MIN(1.D0,-2.*DF/DF1)
        F1=F
        IF(X2.LE.0.0) X2=1
        CALL LINMIN(X1,X2,F1,DF1,REPS,AEPS,IER1,FCN,WK,X,N,NUM)
        IF(IER1.GT.0) IER=IER1
      !If line search fails, then quit
        IF(IER1.GT.100) RETURN

      !The convergence test
        QC=.TRUE.
        DO 2800 I=1,N
          X(I)=WK(N+I)
          WK(N+I)=X1*WK(I)
          IF(ABS(WK(N+I)).GT.MAX(REPS*ABS(X(I)),AEPS)) QC=.FALSE.
          GI=WK(N2+I)
          WK(N2+I)=WK(N2+I)-G(I)
          G(I)=GI
2800    CONTINUE
      !It is possible to apply convergence check on Function value using DF
      !instead of X(I)
        DF=F-F1
        F=F1
        IF(QC) THEN
          IF(ABS(H2/H1).GT.1.3D0) IER=53
          RETURN
        ENDIF

      !Update the matrix using BFGS formula
        DO 3200 I=1,N
          WK(I)=0.0
          DO 3200 J=1,N
3200    WK(I)=WK(I)+H(I,J)*WK(N2+J)
        GHG=0.0
        DG=0.0
        DO 3400 I=1,N
          DG=DG+WK(N2+I)*WK(N+I)
3400    GHG=GHG+WK(I)*WK(N2+I)
        R1=(1.+GHG/DG)/DG
        DO 3600 J=1,N
          DO 3600 I=1,N
3600   H(I,J)=H(I,J)+R1*WK(N+I)*WK(N+J)-(WK(N+I)*WK(J)+WK(I)*WK(N+J))/DG

4000  CONTINUE

      !Iteration fails to converge
      IER=526
      END SUBROUTINE BFGS



      !*************************************************************************
      !TO PERFORM A LINE SEARCH FOR MINIMUM OF SEVERAL VARIABLES AS REQUIRED BY
      !QUASI-NEWTON METHODS. THIS ROUTINE SHOULD NOT BE USED FOR ANY OTHER
      !PURPOSE.
      !*************************************************************************
      !
      !X1 : (input/output) Starting value for the line search. After execution
      !	it should contain the distance to minimiser along the line
      !X2 : (input/output) Initial estimate for the minimum along the line.
      !	 This value will be modified by the subroutine.
      !F1 : (input/output) The function value at X1, this value must be supplied
      !DF1 : (output) The first derivative along the search direction at X1
      !REPS : (input) Required relative accuracy
      !AEPS : (input) Required absolute accuracy,
      !	These criterion is only used to terminate line search under
      !	certain conditions and not generally applicable to the line
      !	search. These values should be same as what is used in subroutine BFGS
      !IER : (output) Error parameter, IER=0 implies successful execution
      !	IER=54 implies that subroutine failed to find acceptable point
      !		but the function value at last point is less than
      !		that at beginning.
      !	IER=55 implies that subroutine failed to find acceptable point
      !		even though the interval has been reduced to required accuracy
      !		but the function value at last point is less than
      !		that at beginning.
      !	IER=527 implies that iteration failed to find any point where
      !		the function value is less than the starting value
      !	IER=528 implies that iteration failed to find any point where
      !		the function value is less than the starting value
      !		even though the interval has been reduced to required accuracy
      !F : (input) Name of the subroutine to calculate the function value
      !     and its derivatives
      !V : (input/output) Real array of length 3N. First N element specify
      !	the direction in which minimisation is required. The next
      !	N elements will contain the coordinates of the minimiser
      !	found by LINMIN. The last 3N elements will contain the gradient
      !	vector at the minimiser.
      !XI : (input) Real array of length N containing the coordinates of
      !	starting point for line search
      !N : (input) Number of variables in the function to minimised
      !NUM : (output) Integer variable to keep count of the number of
      !	function evaluations used so far
      !
      !SUBROUTINE F(N,X,FX,G) to calculate the required function, must be supplied
      !	by the user. Here N is the number of variables, FX is the
      !	function value at X and G is the gradient vector. X and G
      !	are real arrays of length N.
      !
      !Required routines :  FLNM, F
      !
      !*************************************************************************

      SUBROUTINE LINMIN(X1,X2,F1,DF1,REPS,AEPS,IER,F,V,XI,N,NUM)
      IMPLICIT NONE

      LOGICAL :: QB

      INTEGER, PARAMETER :: NIT=15
      INTEGER :: N, IER, NUM
      INTEGER :: I

      REAL*8, PARAMETER :: RHO=0.01D0, SIGMA=0.1D0
      REAL*8, PARAMETER :: T1=9.0, T2=0.9D0, T3=0.5D0
      REAL*8 :: REPS, AEPS
      REAL*8 :: X0, X1, X2, XA, DX, DX1
      REAL*8 :: F0, F1, F2, FA, FC, DF0, DF1, DF2, DFA, DF12
      REAL*8 :: R, R1, R2
      REAL*8 :: V(3*N), XI(N)

      EXTERNAL :: F


      IER=0
      !Select the bracketing phase
      QB=.FALSE.
      F2=FLNM(F,X2,DF2,V,XI,N,NUM)
      DX1=X2-X1
      F0=F1
      DF0=DF1
      X0=X1

      DO 2000 I=1,NIT
        FC=F0+DF0*RHO*(X2-X0)
        IF(ABS(DF2).LE.-SIGMA*DF0.AND.F2.LE.FC) THEN
      !Found an acceptable point
          X1=X2
          F1=F2
          DF1=DF2
          RETURN
        ENDIF
      !Test for bracketing
        IF(.NOT.QB) THEN
          IF(F2.GT.FC.OR.F2.GT.F1.OR.DF2.GE.0) QB=.TRUE.
        ENDIF

      !Hermite cubic interpolation
        DF12=(F2-F1)/DX1
        R=2.*DF2+DF1-3.*DF12
        R1=(3.*DF12-DF2-DF1)**2-DF1*DF2
        DX=0.0
        IF(R1.GT.0.0) THEN
          R1=SIGN(SQRT(R1),R)
          R=R+R1
          DX=-DF2*DX1/R
        ELSE
      !try parabolic interpolation
          R2=2.*(DF12-DF2)
          IF(R2.NE.0.0) DX=DX1*DF2/R2
        ENDIF

        IF(QB) THEN
      !Minimum is bracketed and hence improve on the bracket
          IF(DX.LT.-T2*DX1) DX=-T2*DX1
          IF(DX.GT.-T3*DX1) DX=-T3*DX1
          XA=X2+DX
          FA=FLNM(F,XA,DFA,V,XI,N,NUM)
          FC=F0+DF0*RHO*(XA-X0)

          IF(ABS(DFA).LE.-SIGMA*DF0.AND.FA.LE.FC) THEN
      !The new point is acceptable
            X1=XA
            F1=FA
            DF1=DFA
            RETURN
          ELSE IF(FA.GT.FC.OR.FA.GE.F1.OR.DFA.GT.0.0) THEN
            X2=XA
            F2=FA
            DF2=DFA
          ELSE
            X1=XA
            F1=FA
            DF1=DFA
          ENDIF

          DX1=X2-X1
          IF(ABS(DX1).LT.MAX(REPS*ABS(X2),AEPS)) THEN
      !If the interval is too small, then quit
            IER=528
            IF(F2.LE.F0) THEN
      !Accept the last point in any case
              X1=X2
              F1=F2
              DF1=DF2
              IER=55
            ENDIF
            RETURN
          ENDIF
        ELSE
      !Minimum hasn't been bracketed, choose the point further down.
          IF(DX.LT.X2-X1.AND.DX.GT.0) DX=X2-X1
          IF(DX.GT.T1*(X2-X1).OR.DX.LE.0.0) DX=T1*(X2-X1)
          X1=X2
          X2=X2+DX
          DX1=DX
          F1=F2
          DF1=DF2
          F2=FLNM(F,X2,DF2,V,XI,N,NUM)
        ENDIF
2000  CONTINUE

      !Iteration has failed to find an acceptable point
      IF(F2.LE.F0) THEN
      !accept this point if function value is smaller
        F1=F2
        X1=X2
        DF1=DF2
        IER=54
        RETURN
      ELSE IF(F1.LE.F0.AND.X1.NE.X0) THEN
        IER=54
        RETURN
      ENDIF

      !No acceptable point found
      IER=527
      END SUBROUTINE LINMIN



      !*************************************************************************
      !FUNCTION ROUTINE TO CALCULATE THE FUNCTION VALUE AND ITS DERIVATIVE
      !AS REQUIRED FOR LINE SEARCH.
      !*************************************************************************
      !
      !FCN : (input) Name of subroutine to calculate the required function
      !X : (input) Parameter along the line to specify the point where
      !	function evaluation is required
      !DF : (output) First derivative of function along the line at X
      !V : (input/output) Real array of length 3N, first N elements specify the
      !	direction of line search. Next N elements will contain the
      !	coordinates of the point at which function is evaluated,
      !	while the last N elements contain the gradient vector at the point
      !X0 : (input) Real array of length N, containing the coordinates
      !	of starting point for line search
      !N : (input) Number of variables in the function to be minimised
      !NUM : (input/output) Integer variable to keep count of function evaluations
      !
      !SUBROUTINE FCN(N,X,FX,G) to calculate the required function, must be supplied
      !	by the user. Here N is the number of variables, FX is the
      !	function value at X and G is the gradient vector. X and G
      !	are real arrays of length N.
      !
      !Required routines : FCN
      !
      !*************************************************************************

      FUNCTION FLNM(FCN,X,DF,V,X0,N,NUM)
      IMPLICIT NONE

      INTEGER :: N, NUM
      INTEGER :: I, N2

      REAL*8 :: FLNM, X, DF
      REAL*8 :: V(3*N),X0(N)

      EXTERNAL :: FCN


      NUM=NUM+1
      N2=2*N
      !The coordinates of the required point
      DO 1000 I=1,N
1000  V(N+I)=X0(I)+V(I)*X

      CALL FCN(N,V(N+1),FLNM,V(N2+1))
      !The first derivative along the search direction
      DF=0.0
      DO 2000 I=1,N
2000  DF=DF+V(I)*V(N2+I)
      END FUNCTION FLNM



      !*************************************************************************
      !TO COMPUTE THE FUNCTION TO BE MINIMIZED
      !*************************************************************************
      !
      !m : (input) Number of parameters
      !x : (input) Real array of length m containing the parameters
      !f : (output) The function value at x
      !g : (output) Real array of length m containing the gradient vector at x
      !
      !*************************************************************************

      SUBROUTINE FCN_SD(m,x,f,g)
      IMPLICIT NONE

      INTEGER :: m
      INTEGER :: i, i0, j, k

      REAL*8 :: f
      REAL*8 :: x(m), g(m), dif(num_of_dif2)
      REAL*8 :: fit_func, dpoly
      REAL*8 :: nu, piNu, factor, tmp


      ! Difference between observation and model
      i0 = npoly_sd
      DO i = 1, num_of_dif2
        nu = MU * freqDif2(i,3)
        piNu = 4.d0 * PI * nu
        fit_func = FIT_FUNC_SD(freqDif2(i,3),x)
        dif(i) = freqDif2(i,5) - fit_func
      ENDDO


      f = 0.d0
      g = 0.d0
      DO i = 1, num_of_dif2
        nu = MU * freqDif2(i,3)
        piNu = 4.d0 * PI * nu
        DO j = 1, num_of_dif2

          ! Function value (standard chi-square term)
          f = f + dif(i) * icov(i,j) * dif(j)

          ! Gradient (standard chi-square term)
          factor = - 2.d0 * icov(i,j) * dif(j)
          DO k = 1, i0
            g(k) = g(k) + factor * freqDif2(i,3)**(k-1)
          ENDDO
          g(i0+1) = g(i0+1) + factor * 2.d0 * x(i0+1) * &
                    SIN(piNu * x(i0+2) + x(i0+3))/freqDif2(i,3)**2
          g(i0+2) = g(i0+2) + factor * x(i0+1)**2 * piNu * &
                    COS(piNu * x(i0+2) + x(i0+3))/freqDif2(i,3)**2
          g(i0+3) = g(i0+3) + factor * x(i0+1)**2 * &
                    COS(piNu * x(i0+2) + x(i0+3))/freqDif2(i,3)**2
          g(i0+4) = g(i0+4) + factor * 2.d0 * x(i0+4) * &
                    freqDif2(i,3) * EXP(-(x(i0+5) * nu)**2) * &
                    SIN(piNu * x(i0+6) + x(i0+7))
          g(i0+5) = g(i0+5) - factor * x(i0+4)**2 * freqDif2(i,3) * &
                    2.d0 * x(i0+5) * nu**2 * EXP(-(x(i0+5) * nu)**2) * &
                    SIN(piNu * x(i0+6) + x(i0+7))
          g(i0+6) = g(i0+6) + factor * x(i0+4)**2 * freqDif2(i,3) * &
                    EXP(-(x(i0+5) * nu)**2) * piNu * &
                    COS(piNu * x(i0+6) + x(i0+7))
          g(i0+7) = g(i0+7) + factor * x(i0+4)**2 * &
                    freqDif2(i,3) * EXP(-(x(i0+5) * nu)**2) * &
                    COS(piNu * x(i0+6) + x(i0+7))
        ENDDO
        dpoly = DPOLY_SD(freqDif2(i,3),x)

        ! Function value (regularization term)
        f = f + (regu_param_sd * dpoly)**2

        ! Gradient (regularization term)
        DO j = nderiv_sd+1, npoly_sd
          tmp = 1.d0
          DO k = 1, nderiv_sd
            tmp = tmp * (j-k)
          ENDDO
          g(j) = g(j) + 2.d0 * regu_param_sd**2 * dpoly * &
                 tmp * freqDif2(i,3)**(j-nderiv_sd-1)
        ENDDO
      ENDDO

      END SUBROUTINE FCN_SD



      !*************************************************************************
      !TO COMPUTE THE FITTING FUNCTION
      !*************************************************************************
      !
      !nu : (input) Oscillation frequency of the mode (in muHz)
      !param : (input) Array containing the fitting parameters
      !FIT_FUNC_SD : (output) Value of the fitting function
      !
      !*************************************************************************

      FUNCTION FIT_FUNC_SD(nu,param)
      IMPLICIT NONE

      INTEGER :: i, i0

      REAL*8 :: nu, piNu
      REAL*8 :: param(total_num_of_param_sd)
      REAL*8 :: FIT_FUNC_SD


      ! Smooth term
      FIT_FUNC_SD = 0.d0
      DO i = npoly_sd, 1, -1
        FIT_FUNC_SD = FIT_FUNC_SD * nu + param(i)
      ENDDO

      ! CZ term
      piNu = 4 * PI * MU * nu
      i0 = npoly_sd
      FIT_FUNC_SD = FIT_FUNC_SD + (param(i0+1)/nu)**2 * &
                     SIN(piNu * param(i0+2) + param(i0+3))

      ! He term
      FIT_FUNC_SD = FIT_FUNC_SD + param(i0+4)**2 * nu * &
                     EXP(-(param(i0+5) * MU * nu)**2) * &
                     SIN(piNu * param(i0+6) + param(i0+7))

      END FUNCTION FIT_FUNC_SD



      !*************************************************************************
      !TO COMPUTE THE DERIVATIVE OF THE SMOOTH COMPONENT
      !*************************************************************************
      !
      !nu : (input) Oscillation frequency of the mode (in muHz)
      !param : (input) Array containing the fitting parameters
      !DPOLY_SD : (output) Derivative of the smooth component
      !
      !*************************************************************************

      FUNCTION DPOLY_SD(nu,param)
      IMPLICIT NONE

      INTEGER :: i, j

      REAL*8 :: nu
      REAL*8 :: param(total_num_of_param_sd)
      REAL*8 :: DPOLY_SD, tmp


      ! Check
      DPOLY_SD = 0.d0
      IF (nderiv_sd .GE. npoly_sd) THEN
        RETURN
      ENDIF

      ! Derivative
      DO i = npoly_sd-nderiv_sd, 1, -1
        tmp = 1.d0
        DO j = 1, nderiv_sd
          tmp = tmp * (i + j - 1)
        ENDDO
        DPOLY_SD = DPOLY_SD * nu + tmp * param(i+nderiv_sd)
      ENDDO

      END FUNCTION DPOLY_SD



      !*************************************************************************
      !TO COMPUTE THE REGULARIZATION TERM
      !*************************************************************************
      !
      !param : (input) Array containing fitted parameters.
      !
      !*************************************************************************

      FUNCTION REGU_SD(param)
      IMPLICIT NONE

      INTEGER :: i

      REAL*8 :: param(total_num_of_param_sd)
      REAL*8 :: REGU_SD
      REAL*8 :: dpoly


      ! Check
      REGU_SD = 0.d0
      IF (nderiv_sd .GE. npoly_sd) THEN
        RETURN
      ENDIF

      DO i = 1, num_of_dif2
        dpoly = DPOLY_SD(freqDif2(i,3),param)
        REGU_SD = REGU_SD + (regu_param_sd * dpoly)**2
      ENDDO

      END FUNCTION REGU_SD

      END SUBROUTINE FIT_SD
