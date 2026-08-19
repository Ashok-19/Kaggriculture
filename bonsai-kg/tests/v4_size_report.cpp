#include "../include/bonsai_kg_v4.hpp"
#include "../include/bonsai_kg_v4_io.hpp"
#include <iostream>
int main(){using namespace bonsaikg4;
std::cout<<"Header4="<<sizeof(Header4)<<"\n";
std::cout<<"Node4="<<sizeof(Node4)<<"\n";
std::cout<<"BbDecl4="<<sizeof(BbDecl4)<<"\n";
std::cout<<"ValueRef4="<<sizeof(ValueRef4)<<"\n";
std::cout<<"FeatureArgRef4="<<sizeof(FeatureArgRef4)<<"\n";
std::cout<<"Predicate4="<<sizeof(Predicate4)<<"\n";
std::cout<<"Lifecycle4="<<sizeof(Lifecycle4)<<"\n";
std::cout<<"Effect4="<<sizeof(Effect4)<<"\n";
std::cout<<"NodeState4="<<sizeof(NodeState4)<<"\n";
std::cout<<"RuntimeMeta4="<<sizeof(RuntimeMeta4)<<"\n";
std::cout<<"DeferredIntent4="<<sizeof(DeferredIntent4)<<"\n";
std::cout<<"UnitClaim4="<<sizeof(UnitClaim4)<<"\n";
std::cout<<"MarketClaim4="<<sizeof(MarketClaim4)<<"\n";
std::cout<<"EvalFrame4="<<sizeof(EvalFrame4)<<"\n";
std::cout<<"HaltFrame4="<<sizeof(HaltFrame4)<<"\n";
}
