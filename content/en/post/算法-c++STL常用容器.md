---
date: 2025-03-14T11:00:59-04:00
description: ""
featured_image: "/images/c++STL/lucky.jpg"
tags: ["c++", "algorithm"]
title: "算法 - C++STL常用容器"
---

#### 插入函数总结

| 方法           | 适用容器                           | 作用                       | 性能特性                           |
| :------------- | :--------------------------------- | :------------------------- | :--------------------------------- |
| `push`         | `queue`、`stack`、`priority_queue` | 添加元素到容器末尾或顶部   | 适用于特定容器，性能与容器实现相关 |
| `push_back`    | `vector`、`deque`、`list`          | 添加元素到容器末尾         | 需要拷贝或移动元素                 |
| `emplace`      | `set`、`map`、`unordered_set` 等   | 在容器中直接构造元素       | 避免不必要的拷贝或移动             |
| `emplace_back` | `vector`、`deque`、`list`          | 在容器末尾直接构造元素     | 避免不必要的拷贝或移动             |
| `insert`       | 大多数容器                         | 将元素插入到容器的指定位置 | 需要拷贝或移动元素                 |

<!--more-->

------

&nbsp;

### **1. string 字符串**

+ `getline(cin, s)` ：用于从标准输入（通常是键盘）读取一整行文本。

  1. **读取整行**：会读取从当前光标位置直到遇到换行符（Enter键）的所有内容
  2. **包含空格**：与 `cin >>` 不同，`getline` 会读取空格
  3. **丢弃换行符**：读取的内容不包含最后的换行符

+ ##### `insert()` 函数

  用于在字符串的指定位置插入内容。

  ```c++
  string s = "HelloWorld";
  string t = "Beautiful";
  s.insert(5, t);  // 在位置5插入
  cout << s;  // 输出：HelloBeautifulWorld
  ```

+ ##### `erase()` 函数

  用于删除字符串中的部分内容.

  ```c++
  string s = "HelloBeautifulWorld";
  s.erase(5, 9);  // 删除从位置5开始的9个字符
  cout << s;  // 输出：HelloWorld
  ```

+ ##### `substr()` 函数

  - 返回从位置 `pos` 开始、长度为 `len` 的子字符串。
  - 如果省略 `len`，则返回从 `pos` 开始到字符串末尾的子字符串。

  ```c++
  // 提取从位置0开始的5个字符
  string sub2 = s.substr(0, 5);
  ```

+ ##### `find()` 函数

  用于在字符串中查找子字符串或字符。

  ##### 返回值：

  - 找到则返回首次出现的位置
  - 未找到则返回 `string::npos`

  ```c++
  // 查找子字符串
  size_t found = s.find("World");
  if (found != string::npos) {
    cout << "'World' found at: " << found << endl;  // 输出: 7
  }
  ```

+ ##### `replace()` 

  用于替换字符串中的部分内容。

  ```c++
  string s = "I like apples and oranges.";
  // 替换子字符串
  s.replace(7, 6, "bananas");
  cout << s << endl;  // 输出: I like bananas and oranges.
  ```

```c++
string s;
getline(cin, s);  // 读取一整行输入，存入字符串s中
cout << "你输入的是: " << s << endl;
```

```cpp
#include <string>
using namespace std;

string s = "Hello";
s += " World";          // 拼接

char first_char = str[0];  			// 访问第1个字符（索引为0）'H'
string sub = s.substr(6, 5); 		// substr(起始位置, 截取长度) → "World"
size_t pos = s.find("World");		// 查找子串，返回位置，未找到返回 string::npos

s.length();              // 字符串长度
s.size(); 
s.empty();               // 判断是否为空
s.starts_with("He");     // C++20 判断前缀
s.contains("or");        // C++20 判断是否包含子串
```



##### （3）`to_string`

+ 转为字符串格式

#### 遍历找连续0

```c++
#include <iostream>
#include <string>
using namespace std;

int longestZeroSubstring(const string& s) {
    int max_len = 0;
    int current = 0;
  
    for (char c : s) {
        if (c == '0') {
            current++;
            if (current > max_len) {
                max_len = current;
            }
        } else {
            current = 0;
        }
    }
  
    return max_len;
}

int main() {
    string s;
    cout << "请输入一个字符串: ";
    cin >> s;
    int result = longestZeroSubstring(s);
    cout << "最长连续 '0' 的子串长度是: " << result << endl;
    return 0;
}
```



---

&nbsp;

### **2. **stack 栈

```c++
stack<int> s;
```

1. 压入元素：`push()`

2. 弹出最后添加的元素：`pop()`，不返回该元素

3. 访问栈顶元素：`top()`

4. 检查栈是否为空：`empty()`

5. 获取栈的大小：`size()`

&nbsp;

### 3. vector 动态数组

注意：

```c++
vector<position> tempPath(); // 这是一个函数声明，而不是变量初始化
vector<position> tempPath; // 正确：初始化一个空的 vector
```



```c++
std::vector<int> vec; 												// 创建一个空的 int 类型 vector
std::vector<int> vec(10); 										// 创建一个包含 10 个元素的 vector，初始值为 0
std::vector<int> vec(10, 5); 									// 创建一个包含 10 个元素的 vector，初始值为 5
std::vector<int> vec = {1, 2, 3, 4, 5}; 			// 创建一个包含 5 个元素的 vector
```

```cpp
int first = vec.front(); 								// 访问第一个元素
int last = vec.back();  								// 访问最后一个元素

v.push_back(10);          // 尾部插入 int
v.push_back({x, y});      // 尾部插入 vector

v.pop_back();             // 尾部删除

v.size();                 // 元素数量

for (auto num : v) { /* 遍历 */ }
for (int i=0; i<v.size(); ++i) { v[i]; }
```

---

&nbsp;

### 4. map 有序键值对

```cpp
#include <map>
map<string, int> m;

m["apple"] = 5;         // 插入/修改键值
m.erase("apple");       // 删除键
m.size();               // 键值对数量
m.contains("apple");    // C++20 判断键是否存在

// 遍历（按键升序）
for (auto& [key, val] : m) { 
    cout << key << ": " << val << endl; 
}

// 注意：访问 m["key"] 时若 key 不存在，会自动插入默认值！
```

---

&nbsp;

### 5. set 有序唯一集合

```cpp
#include <set>
set<int> s;

s.insert(3);            // 插入元素（自动去重）
s.erase(3);             // 删除元素
s.contains(3);          // 判断存在性
s.size();               // 元素数量

// 遍历（按升序）
for (auto num : s) { cout << num << " "; }

// 典型用途：排序+去重
vector<int> nums = {2,1,2,3};
set<int> tmp(nums.begin(), nums.end()); // → {1,2,3}
```

---

&nbsp;

### 6. queue 队列

```c++
queue<int> q;						// 初始化

q.push(10);							// 添加元素

cout << q.front() 			// 访问队首
cout << q.back()   			// 访问队尾

q.pop();								// 移除队首元素，
q.empty();							// 检查队列是否为空
std::cout << q.size();  // 获取队列大小
```
#### 6.5 priority_queue 优先队列

`priority_queue` 默认是一个 **最大堆**（Max Heap），即优先级最高的元素（默认是最大的元素）会最先出队。

插入操作是通过 `push()` 或 `emplace()` 来完成的，而不是 `push_back`。

```c++
priority_queue<int> pq; // 定义：默认是最大堆

pq.push(10); // 插入元素 10
pq.push(5);  // 插入元素 5
int topElement = pq.top(); // 获取堆顶元素（最大值）
pq.pop(); // 删除堆顶元素
```

定义最小堆：

```c++
// greater<int> 是一个比较函数，表示元素越小优先级越高
priority_queue<int, vector<int>, greater<int>> minHeap;
```

对于 `priority_queue<pair<int, int>>`，默认情况下，它会按照 `pair` 的**第一个元素**（即 `first`）进行排序。如果第一个元素相同，则会比较第二个元素（即 `second`）

&nbsp;

### 7. pair<int, int>

用于存储两个值（可以是相同类型或不同类型)

- 在输出 `optPath` 时，`path[0]` 和 `path[1]` 的写法是错误的，因为 `path` 是一个 `pair<int, int>`，应该使用 `path.first` 和 `path.second`。

### 8. tuple<int, int, int>

用于存储三个值（可以是相同类型或不同类型)

&nbsp;

1. **map 遍历有序性**：题目若要求无序输出需改用 `unordered_map`（哈希表，O(1)访问）
2. **set 去重原理**：插入时若值已存在则忽略，需去重时优先考虑
3. **复杂度**：
   - vector 尾部操作 O(1)，中间插入 O(n)
   - map/set 增删查 O(log n)

