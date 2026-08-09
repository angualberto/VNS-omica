program verify_nvfortran_cuda
  use cudafor
  implicit none
  integer, parameter :: n = 1048576
  real, allocatable :: host(:)
  real, device, allocatable :: dev(:)
  integer :: i, ierr, device_count

  ierr = cudaGetDeviceCount(device_count)
  if (ierr /= 0 .or. device_count < 1) stop "No CUDA device found"
  allocate(host(n), dev(n))
  host = 0.0
  dev = host

  !$cuf kernel do(1) <<<*,*>>>
  do i = 1, n
    dev(i) = real(i) * 2.0
  end do
  host = dev
  ierr = cudaDeviceSynchronize()
  if (abs(host(n) - real(n) * 2.0) > 0.1) stop "CUDA result validation failed"

  print '(A,I0)', "CUDA devices: ", device_count
  print '(A,F12.1)', "validated_value: ", host(n)
  print '(A)', "CUDA Fortran execution: OK"
end program verify_nvfortran_cuda
