!*******************************************************************************
!
!     SUBROUTINE TO COMPUTE SECOND DIFFERENCES OF OSCILLATION FREQUENCIES
!     --------------------------------------------------------------------------
!
!     AUTHOR NAME     : KULDEEP VERMA
!     EMAIL ADDRESSES : verma@ice.csic.es, kuldeep@phys.au.dk,
!                       kuldeepv89@gmail.com
!     LAST MODIFIED   : 12/01/2022
!
!*******************************************************************************
!
!     freq : (input) Real array containing the oscillation data
!            (l, n, freq(muHz), err(muHz))
!     num_of_n : (input) Integer array containing number of modes for each l
!     num_of_l : (input) Number of harmonic degree
!     num_of_mode : (input) Number of modes
!     num_of_dif2 : (input) Number of second differences
!     dif2 : (output) Real array containing the second differences
!            (l, n, freq(muHz), err(muHz), dif2(muHz), err(muHz))
!
!*******************************************************************************

      SUBROUTINE SD(freq,num_of_n,num_of_l,num_of_mode,num_of_dif2,dif2)
      IMPLICIT NONE

      INTEGER :: num_of_l, num_of_mode, num_of_dif2
      INTEGER :: num_of_n(num_of_l)
      INTEGER :: i, j, k, l

      REAL*8 :: freq(num_of_mode,4)
      REAL*8 :: dif2(num_of_dif2,6)
!f2py intent(in)  :: num_of_l, num_of_mode, num_of_dif2, num_of_n, freq
!f2py intent(out) :: dif2

      !Check for missing radial order
      k = 0
      DO i = 1, num_of_l
        IF (num_of_n(i) .EQ. 0) CYCLE
        k = k + num_of_n(i)
        j = NINT(freq(k,2) - freq(k-num_of_n(i)+1,2) + 1)
        IF (j .NE. num_of_n(i)) THEN
          WRITE (*,'(A30)') 'ERROR: Missing radial order!'
          WRITE (*,'(A10,I2,A6)') 'Check l =', i - 1, 'modes'
          STOP
        ENDIF
      ENDDO

      !Compute second differences (assuming no missing n)
      dif2 = 0.d0 !Initialize second differences
      k = 0       !run over # of modes
      l = 0       !run over # of second differences
      DO i = 1, num_of_l
        IF (num_of_n(i) .EQ. 0) CYCLE
        DO j = 1, num_of_n(i)
          IF (j .EQ. 1 .OR. j .EQ. num_of_n(i)) THEN
            k = k + 1
            CYCLE
          ELSE
            k = k + 1
            l = l + 1
            dif2(l,1) = freq(k,1)
            dif2(l,2) = freq(k,2)
            dif2(l,3) = freq(k,3)
            dif2(l,4) = freq(k,4)
            dif2(l,5) = freq(k-1,3) - 2.d0*freq(k,3) + freq(k+1,3)
            dif2(l,6) = freq(k-1,4)**2 + freq(k+1,4)**2
            dif2(l,6) = SQRT(dif2(l,6) + 4.d0*freq(k,4)**2)
          ENDIF
        ENDDO
      ENDDO

      END SUBROUTINE SD
