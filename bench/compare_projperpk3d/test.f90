! sum.f90
! Performs summations using in a loop using EXIT statement
! Saves input information and the summation in a data file

program summation
implicit none
integer, parameter :: N0=128, N1=128, N2=65, N=1000
double precision, allocatable :: vx_fft(:,:,:,:), vy_fft(:,:,:,:), vz_fft(:,:,:,:)
double precision, allocatable :: Kx(:,:,:), Ky(:,:,:), Kz(:,:,:)
double precision, allocatable :: inv_K_square_nozero(:,:,:)
double precision, allocatable :: res(:,:,:,:,:)
real :: start, finish, cumtime
integer :: i

allocate(vx_fft(2, N2, N1, N0), vy_fft(2, N2, N1, N0), vz_fft(2, N2, N1, N0))
allocate(Kx(N2, N1, N0), Ky(N2, N1, N0), Kz(N2, N1, N0))
allocate(inv_K_square_nozero(N2, N1, N0),res(2, 3, N2, N1, N0))

call random_number(vx_fft)
call random_number(vy_fft)
call random_number(vz_fft)
call random_number(Kx)
call random_number(Ky)
call random_number(Kz)
call random_number(inv_K_square_nozero)

cumtime = 0

print*, "This program make some calculations."
do i = 1, N
 call cpu_time(start)
 call proj(res, vx_fft, vy_fft, vz_fft, Kx, Ky, Kz, inv_K_square_nozero, N0, N1, N2)
 call cpu_time(finish)
 cumtime = cumtime + finish - start
enddo
print '("Mean Time = ",f6.3," ms")',1000*cumtime/N




deallocate(vx_fft, vy_fft, vz_fft)
deallocate(Kx, Ky, Kz)
deallocate(inv_K_square_nozero,res)
end program


subroutine proj(res, vx_fft, vy_fft, vz_fft, Kx, Ky, Kz, inv_K_square_nozero, N0, N1, N2)

   implicit none

!  Input/Output
   integer, intent(in) :: N0, N1, N2
   double precision, intent(in) :: vx_fft(2, N2, N1, N0), vy_fft(2, N2, N1, N0), vz_fft(2, N2, N1, N0)
   double precision, intent(in) :: Kx(N2, N1, N0), Ky(N2, N1, N0), Kz(N2, N1, N0)
   double precision, intent(in) :: inv_K_square_nozero(N2, N1, N0)
   double precision, intent(out) :: res(2, 3, N2, N1, N0)

!  Locals
   double precision, allocatable :: tmp(:, :, :, :)
   integer:: i, j, k


   allocate(tmp(2, N2, N1, N0))
   call calc_tmp(tmp, vx_fft, vy_fft, vz_fft, Kx, Ky, Kz, inv_K_square_nozero, N0, N1, N2)
   do k = 1, N0
    do j = 1, N1
     do i = 1, N2
      res(1:2,1,i,j,k) = vx_fft(1:2,i,j,k) - Kx(i,j,k) * tmp(1:2,i,j,k)
      res(1:2,2,i,j,k) = vy_fft(1:2,i,j,k) - Ky(i,j,k) * tmp(1:2,i,j,k)
      res(1:2,3,i,j,k) = vz_fft(1:2,i,j,k) - Kz(i,j,k) * tmp(1:2,i,j,k)
     enddo
    enddo
   enddo
   deallocate(tmp)

 contains
   subroutine calc_tmp(tmp, vx_fft, vy_fft, vz_fft, Kx, Ky, Kz, inv_K_square_nozero, N0, N1, N2)
   integer, intent(in) :: N0, N1, N2
      double precision, intent(in) :: vx_fft(2,N2, N1, N0), vy_fft(2,N2, N1, N0), vz_fft(2,N2, N1, N0)
      double precision, intent(in) :: Kx(N2, N1, N0), Ky(N2, N1, N0), Kz(N2, N1, N0)
      double precision, intent(in) :: inv_K_square_nozero(N2, N1, N0)
      double precision, intent(out) :: tmp(2,N2, N1, N0)

      integer:: i, j, k
      do k = 1, N0
       do j = 1, N1
        do i = 1, N2
         tmp(1:2, i,j,k) = (Kx(i,j,k) * vx_fft(1:2,i,j,k) + Ky(i,j,k) * vy_fft(1:2,i,j,k) + &
                            Kz(i,j,k) * vz_fft(1:2,i,j,k)) * inv_K_square_nozero(i,j,k)
        enddo
       enddo
      enddo

   end subroutine calc_tmp
end subroutine proj
