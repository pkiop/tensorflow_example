#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <algorithm>
#include <fstream>
#include <string>
#include <cstring>
#include <map>
#include <set>
 
#define pii pair<int, int>
#define pli pair<long, int>
#define mii map<int, int>
#define msi map<string, int>
 
typedef unsigned long long ull;
typedef long long ll;
 
using namespace std;
 
#define GO
#define DEBUG
 
/*
 
3
4
10 20
30 40
50 60
70 80
2
1 3
2 200
3
10 100
20 80
30 50   // T : 테스트케이스 수
// N : 돌아가야 할 학생들의 수
// 10 : 현재 방, 20 : 돌아갈 방
 
 
 
// 두번째 테스트케이스의 N
 
 
 
 
*/
 
 
int room[201];
int main(void) {
 
    int t;
    int test = 0;
    cin >> t;
    while (t--) {
        memset(room, 0, sizeof(room));
        int n;
        int maxroom = 0;
        cin >> n;
        for (int i = 0; i < n; ++i) {
            int a, b;
            cin >> a >> b;
            if(a > b)
                swap(a, b);
            int inc = 0;
            maxroom = b;
            for (int j = a; j <= b; ++j) {
                if(inc != j/2){
                    room[j/2]++;
                    inc = j/2;
                }
            }
        }
        int ans = 1;
        for (int i = 1; i <= maxroom; ++i) {
            if (room[i] == 0) {
                continue;
            }
            ans = max(ans, room[i]);
        }
        test++;
        cout << '#' << test << ' ' << ans << endl;
    }
 
 
 
    return 0;
}