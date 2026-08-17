#include "../cuda/kag_adapter.cuh"
#include <type_traits>
#include <utility>

template<class T,class=void>struct has_obs:std::false_type{};
template<class T>struct has_obs<T,std::void_t<decltype(std::declval<T>().obs)>>:std::true_type{};
template<class T,class=void>struct has_obs_private_name:std::false_type{};
template<class T>struct has_obs_private_name<T,std::void_t<decltype(std::declval<T>().obs_)>>:std::true_type{};
static_assert(!has_obs<bonsaikg::CudaObservation>::value,"CudaObservation must not expose .obs");
static_assert(!has_obs_private_name<bonsaikg::CudaObservation>::value,"CudaObservation backing ObservationView must be inaccessible");
int main(){return 0;}
