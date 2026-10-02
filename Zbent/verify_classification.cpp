// Exact exhaustive check: g++ -O3 -std=c++17 verify_classification.cpp -o verify_classification
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
static void walsh(std::vector<int>& a){
 for(int s=1;s<(int)a.size();s*=2)for(int b=0;b<(int)a.size();b+=2*s)
 for(int j=0;j<s;j++){int x=a[b+j],y=a[b+j+s];a[b+j]=x+y;a[b+j+s]=x-y;}
}
int main(){for(int n:{2,4}){
 const int N=1<<n, scale=1<<(n/2); uint32_t total=1;
 std::vector<uint32_t> pow3(N);for(int i=0;i<N;i++){pow3[i]=total;total*=3;}
 std::vector<std::vector<int>> bent;
 for(uint32_t mask=0;mask<(1u<<N);mask++){
  std::vector<int>a(N);for(int i=0;i<N;i++)a[i]=(mask>>i&1)?1:-1;
  auto t=a;walsh(t);bool ok=true;for(int x:t)if(x!=scale&&x!=-scale){ok=false;break;}
  if(ok)bent.push_back(a);
 }
 std::vector<unsigned char> averages(total,0);uint32_t num_average=0;
 for(size_t i=0;i<bent.size();i++)for(size_t j=i;j<bent.size();j++){
  uint32_t code=0;for(int x=0;x<N;x++)code+=((bent[i][x]+bent[j][x])/2+1)*pow3[x];
  if(!averages[code]){averages[code]=1;num_average++;}
 }
 // Independent ternary enumeration; digits are f(x)+1. Maintain Hf exactly.
 std::vector<int>digits(N,0),t(N,0);t[0]=-N;
 std::vector<std::vector<int>>cols(N,std::vector<int>(N));
 for(int x=0;x<N;x++)for(int u=0;u<N;u++)cols[x][u]=(__builtin_parity((unsigned)(x&u)))?-1:1;
 uint32_t count=0;std::vector<uint32_t>dist(N+1,0);
 for(uint32_t code=0;code<total;code++){
  bool ok=true;for(int v:t)if(v!=0&&v!=scale&&v!=-scale){ok=false;break;}
  if(ok){count++;int supp=0;for(int d:digits)supp+=(d!=1);dist[supp]++;
   if(!averages[code]){std::cerr<<"Non-splitting code "<<code<<"\n";return 1;}}
  if(code+1<total)for(int x=0;x<N;x++){
   int change=digits[x]<2?1:-2;digits[x]=digits[x]<2?digits[x]+1:0;
   for(int u=0;u<N;u++)t[u]+=change*cols[x][u];if(change==1)break;
  }
 }
 assert(count==num_average);assert(count==(n==2?33u:213249u));assert(bent.size()==(n==2?8u:896u));
 std::cout<<"n="<<n<<" ternary="<<total<<" level_one="<<count<<" bent="<<bent.size()<<" averages="<<num_average<<"\n";
 for(int s=0;s<=N;s++)if(dist[s])std::cout<<s<<":"<<dist[s]<<" ";std::cout<<"\n";
}}
