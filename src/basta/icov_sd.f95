!*******************************************************************************
!
!     SUBROUTINES TO COMPUTE INVERSE COVARIANCE MATRIX FOR SECOND DIFFERENCE
!     --------------------------------------------------------------------------
!
!     AUTHOR NAME     : KULDEEP VERMA
!     EMAIL ADDRESSES : verma@ice.csic.es, kuldeep@phys.au.dk,
!                       kuldeepv89@gmail.com
!     LAST MODIFIED   : 12/01/2022
!
!*******************************************************************************
!
!     num_of_l : (input) Number of harmonic degree
!     num_of_n : (input) Integer array containing number of frequencies per l
!     freq : (input) Real array containing the observed oscillation data
!            (l, n, freq(muHz), err(muHz))
!     num_of_dif2 : (input) Number of second differences
!     icov : (output) Real array containing the inverse covariance matrix
!
!*******************************************************************************

      SUBROUTINE ICOV_SD(num_of_l,num_of_n,freq,num_of_dif2,icov)
      IMPLICIT nONE

      INTEGER :: num_of_l, num_of_mode, num_of_dif2
      INTEGER :: num_of_n(:), iwk(num_of_dif2)
      INTEGER :: i, j, k, l, ier

      REAL*8 :: freq(:, :)
      REAL*8 :: jacob(num_of_dif2,size(freq,1))
      REAL*8 :: cov(num_of_dif2,num_of_dif2)
      REAL*8 :: icov(num_of_dif2,num_of_dif2)
!f2py intent(in) :: num_of_l, num_of_n, freq, num_of_dif2
!f2py intent(out) :: icov

      !Jacobian for the second differences
      jacob = 0.d0
      k = 0
      l = 0
      DO i = 1, num_of_l
        IF (num_of_n(i) .EQ. 0) CYCLE
        DO j = 1, num_of_n(i)-2
          k = k + 1
          l = l + 1
          jacob(l,k) = 1.d0
          jacob(l,k+1) = -2.d0
          jacob(l,k+2) = 1.d0
        ENDDO
        k = k + 2
      ENDDO

      !Covariance matrix for the second differences
      num_of_mode = size(freq, 1)
      DO i = 1, num_of_dif2
        DO j = 1, num_of_dif2
          cov(i,j) = 0.d0
          DO k = 1, num_of_mode
            cov(i,j) = cov(i,j) + jacob(i,k) * freq(k,4)**2 * jacob(j,k)
          ENDDO
        ENDDO
      ENDDO

      !Inverse of the covariance matrix
      icov = 0.d0
      CALL MATINV(num_of_dif2,num_of_dif2,cov,&
           icov(1:num_of_dif2,1:num_of_dif2),iwk,ier)
      IF (ier .NE. 0) THEN
        WRITE (*,'(A)') 'ERROR: Failed to invert covariance matrix!'
        WRITE (*,'(A25,I10)') 'Error parameter =', ier
        STOP
      ENDIF

      END SUBROUTINE ICOV_SD



      !*************************************************************************
      !TO CALCULATE INVERSE OF A SQUARE MATRIX.
      !*************************************************************************
      !
      !N : (input) Order of matrix
      !IA : (input) The first dimension of arrays A and AI as specified
      !	in the calling program
      !A : (input) Real array of length IA*N containing the matrix
      !AI : (output) Real array of length IA*N which will contain the
      !	calculated inverse of A
      !IWK : (output) Integer array of length N used as scratch space
      !IER : (output) The error parameter, IER=0 implies successful execution
      !	Nonzero values may be set by subroutine GAUELM
      !It is possible to use CROUT instead of GAUELM for calculating
      !the triangular decomposition, but in that case an extra real scratch
      !array of size N will be required.
      !
      !Required routines : GAUELM
      !
      !*************************************************************************

      SUBROUTINE MATINV(N,IA,A,AI,IWK,IER)
      IMPLICIT nONE

      INTEGER :: N, IA, IER, IFLG, NUM, I, J
      INTEGER :: IWK(N)

      REAL*8 :: DET
      REAL*8 :: A(IA,N), AI(IA,N)

      DO 1000 I=1,N
        DO 800 J=1,N
          AI(J,I)=0.0
800     CONTINUE
        AI(I,I)=1.0
1000  CONTINUE

      NUM=N
      IFLG=0
      CALL GAUELM(N,NUM,A,AI,DET,IWK,IA,IER,IFLG)

      END SUBROUTINE MATINV



      !*************************************************************************
      !SOLUTION OF A SYSTEM OF LINEAR EQUATIONS USING GAUSSIAN ELIMINATION
      !WITH PARTIAL PIVOTING.
      !*************************************************************************
      !
      !N : (input) Number of equations to be solved
      !NUM : (input) Number of different sets (each with N equations) of
      !        equations to be solved
      !A : (input/output) The matrix of coefficient of size LJ*N
      !        A(I,J) is the coefficient of x_J in Ith equation
      !     	at output it will contain the triangular decomposition
      !X : (input/output) The matrix containing right hand sides (size LJ*NUM)
      !        X(I,J) is the Ith element of Jth right hand side
      !     	at output it will contain the solutions
      !DET : (output) The determinant of the matrix
      !INC : (output) Integer array of length N containing information about
      !	interchanges performed during elimination
      !LJ : (input) First dimension of arrays A and X in calling program
      !IER : (output) Error flag, IER=0 signifies successful execution
      !	IER=101 implies (N.LE.0 or N.GT.LJ)
      !	IER=121 implies some pivot turned out to be zero and hence
      !		matrix must be nearly singular
      !IFLG : (input) Integer parameter to specify the type of computation required
      !	If IFLG.LE.0, both elimination and solution are
      !		done and IFLG is set to 2
      !	If IFLG=1, only elimination is done and IFLG is set to 2
      !	If IFLG.GE.2 only solution is calculated, the triangular
      !		decomposition should have been calculated earlier
      !
      !Required routines : None
      !
      !*************************************************************************

      SUBROUTINE GAUELM(N,NUM,A,X,DET,INC,LJ,IER,IFLG)
      IMPLICIT nONE

      INTEGER :: N, NUM, LJ, IER, IFLG, KM, L1, J, K, L
      INTEGER :: INC(N)

      REAL*8 :: DET, R1, T1
      REAL*8 :: A(LJ,N), X(LJ,NUM)

      IF(N.LE.0.OR.N.GT.LJ) THEN
        IER=101
        RETURN
      ENDIF

      IER=121
      IF(IFLG.LE.1) THEN
      !Perform elimination

        DET=1.0
        DO 2600 K=1,N-1
      !Find the maximum element in the Kth column
          R1=0.0
          KM=K
          DO 2200 L=K,N
            IF(ABS(A(L,K)).GT.R1) THEN
              R1=ABS(A(L,K))
              KM=L
            ENDIF
2200      CONTINUE

          INC(K)=KM
          IF(KM.NE.K) THEN
      !Interchange the rows if needed
            DO 2300 L=K,N
              T1=A(K,L)
              A(K,L)=A(KM,L)
2300        A(KM,L)=T1
            DET=-DET
          ENDIF

          DET=DET*A(K,K)
          IF(A(K,K).EQ.0.0) RETURN
      !To check for singular or nearly singular matrices replace this
      !statement by, where REPS is approximately \hcross*Max(A(I,J))
      !    IF(ABS(A(K,K)).LT.REPS) RETURN
          DO 2500 L=K+1,N
            A(L,K)=A(L,K)/A(K,K)
            DO 2500 L1=K+1,N
2500      A(L,L1)=A(L,L1)-A(L,K)*A(K,L1)
2600    CONTINUE
        DET=DET*A(N,N)
        INC(N)=N
      !If pivot is zero then return, IER has been set to 121
        IF(A(N,N).EQ.0.0) RETURN
      !To check for singular or nearly singular matrices replace this
      !statement by, where REPS is approximately \hcross*Max(A(I,J))
      !    IF(ABS(A(N,N)).LT.REPS) RETURN

        IER=0
        IF(IFLG.EQ.1) THEN
          IFLG=2
          RETURN
        ENDIF
        IFLG=2
      ENDIF

      IER=0
      !Solution for the NUM different right-hand sides
      DO 5000 J=1,NUM
        DO 3000 K=1,N-1
      !Forward substitution
          IF(K.NE.INC(K)) THEN
            T1=X(K,J)
            X(K,J)=X(INC(K),J)
            X(INC(K),J)=T1
          ENDIF
          DO 3000 L=K+1,N
3000    X(L,J)=X(L,J)-A(L,K)*X(K,J)

      !back-substitution
        X(N,J)=X(N,J)/A(N,N)
        DO 3300 K=N-1,1,-1
          DO 3200 L=N,K+1,-1
3200      X(K,J)=X(K,J)-X(L,J)*A(K,L)
3300    X(K,J)=X(K,J)/A(K,K)
5000  CONTINUE

      END SUBROUTINE GAUELM
