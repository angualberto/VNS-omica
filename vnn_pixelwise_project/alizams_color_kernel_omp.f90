subroutine alizams_metrics_omp(gray, mask, lut_bin, lut_hue, lut_sat, &
                               lut_val, lut_weight, n, nlut, freq, &
                               hue_mean, sat_mean, val_mean)
  implicit none
  integer, intent(in) :: n, nlut
  real(kind=4), intent(in) :: gray(n)
  integer(kind=4), intent(in) :: mask(n), lut_bin(nlut)
  real(kind=8), intent(in) :: lut_hue(nlut), lut_sat(nlut), lut_val(nlut), lut_weight(nlut)
  real(kind=8), intent(out) :: freq(8), hue_mean, sat_mean, val_mean
  integer :: i, idx, color_bin
  real(kind=8) :: intensity, weight, total_weight, sum_hue, sum_sat, sum_val

  freq = 0.0d0
  total_weight = 0.0d0
  sum_hue = 0.0d0
  sum_sat = 0.0d0
  sum_val = 0.0d0

  !$omp parallel do default(none) private(i,idx,color_bin,intensity,weight) &
  !$omp shared(gray,mask,lut_bin,lut_hue,lut_sat,lut_val,lut_weight,n,nlut) &
  !$omp reduction(+:freq,total_weight,sum_hue,sum_sat,sum_val) schedule(static)
  do i = 1, n
    if (mask(i) /= 0) then
      intensity = min(1.0d0, max(0.0d0, dble(gray(i))))
      idx = min(nlut, max(1, int(intensity * dble(nlut)) + 1))
      color_bin = lut_bin(idx)
      weight = lut_weight(idx)
      freq(color_bin) = freq(color_bin) + weight
      total_weight = total_weight + weight
      sum_hue = sum_hue + weight * lut_hue(idx)
      sum_sat = sum_sat + weight * lut_sat(idx)
      sum_val = sum_val + weight * lut_val(idx)
    end if
  end do
  !$omp end parallel do

  if (total_weight > 0.0d0) then
    freq = freq / total_weight
    hue_mean = sum_hue / total_weight
    sat_mean = sum_sat / total_weight
    val_mean = sum_val / total_weight
  else
    hue_mean = 0.0d0
    sat_mean = 0.0d0
    val_mean = 0.0d0
  end if
end subroutine alizams_metrics_omp
