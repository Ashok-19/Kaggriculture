#include "../cpp/kag_adapter.hpp"
#include <type_traits>
#include <utility>

template<class T,class=void>struct has_sim:std::false_type{};
template<class T>struct has_sim<T,std::void_t<decltype(std::declval<T>().sim)>>:std::true_type{};
template<class T,class=void>struct has_player:std::false_type{};
template<class T>struct has_player<T,std::void_t<decltype(std::declval<T>().player)>>:std::true_type{};
template<class T,class=void>struct has_sim_private_name:std::false_type{};
template<class T>struct has_sim_private_name<T,std::void_t<decltype(std::declval<T>().sim_)>>:std::true_type{};
template<class T,class=void>struct has_player_private_name:std::false_type{};
template<class T>struct has_player_private_name<T,std::void_t<decltype(std::declval<T>().player_)>>:std::true_type{};

static_assert(!has_sim<bonsaikg::CppObservation>::value,"CppObservation must not expose .sim");
static_assert(!has_player<bonsaikg::CppObservation>::value,"CppObservation must not expose .player");
static_assert(!has_sim_private_name<bonsaikg::CppObservation>::value,"CppObservation backing Sim must be inaccessible");
static_assert(!has_player_private_name<bonsaikg::CppObservation>::value,"CppObservation player must be inaccessible");
int main(){return 0;}
