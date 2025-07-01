---
date: 2025-02-13T11:00:59-04:00
description: "std::unordered_map 是 C++ 标准库中的一个关联容器，基于哈希表实现，用于存储键值对。"
featured_image: "/images/algorithm/hash.jpg"
tags: ["c++", algorithm"]
title: "unordered_map"
---

|                    | `unordered_map`                     | `unordered_set`                     |
| :----------------- | :---------------------------------- | :---------------------------------- |
| **存储内容**       | 键值对 (`std::pair<const Key, T>`)  | 仅存储键 (`Key`)                    |
| **是否允许重复键** | 键唯一（重复键会覆盖）              | 键唯一（重复插入无效）              |
| **主要用途**       | 快速通过键**查找值**                | 快速判断**键是否存在**              |
| **典型操作**       | `map[key] = value`、`map.find(key)` | `set.insert(key)`、`set.count(key)` |
| **内存占用**       | 更高（需存储键和值）                | 更低（仅存储键）                    |



### 1. 创建 `unordered_map` 对象

```cpp
#include <unordered_map>
#include <string>

int main() {
    // 默认构造函数
    std::unordered_map<std::string, int> map1;

    // 初始化列表构造函数
    std::unordered_map<std::string, int> map2 = {{"apple", 1}, {"banana", 2}};

    // 拷贝构造函数
    std::unordered_map<std::string, int> map3(map2);

    return 0;
}
```

&nbsp;

### 2. 插入

1. `insert`

   插入一个键值对

   ```cpp
   map1.insert({"orange", 3});
   map1.insert(std::make_pair("grape", 4));
   ```




2. `operator[]`

   通过键插入或访问值。**如果键不存在，会插入一个默认值**。

   ```c++
   map1["apple"] = 10; // 插入或修改
   int value = map1["apple"]; // 访问
   ```




&nbsp;

### 3. **访问元素**
1. `at`

   访问指定键的值，如果键不存在会抛出 `std::out_of_range` 异常。

   ```cpp
   int value = map1.at("apple");
   ```




2. `operator[]`

   访问或插入指定键的值。

   ```cpp
   int value = map1["apple"];
   ```

<!--more-->

&nbsp;

### 4. **删除元素**
1. `erase`

   删除指定键的元素。

   ```c++
   map1.erase("apple"); // 删除键为 "apple" 的元素
   ```

   

2. 清空所有元素

   ```c++
   map1.clear();
   ```

&nbsp;

### 5. **查找元素**
1. `find`

   查找指定键的元素，返回迭代器。如果未找到，返回 `end()`。

   ```c++
   auto it = map1.find("apple");
   if (it != map1.end()) {
       std::cout << "Found: " << it->first << " -> " << it->second << std::endl;
   } else {
       std::cout << "Not found" << std::endl;
   }
   ```



2. `count`

   返回指定键的元素个数（对于 `unordered_map`，只能是 0 或 1）。

   ```c++
   if (map1.count("apple") > 0) {
       std::cout << "Key exists" << std::endl;
   }
   ```



&nbsp;

### 6. **容量相关**
1. `size`

   返回容器中元素的数量。

   ```c++
   std::cout << "Size: " << map1.size() << std::endl;
   ```



2. `empty`

   检查容器是否为空。

   ```c++
   if (map1.empty()) {
       std::cout << "Map is empty" << std::endl;
   }
   ```

   



&nbsp;

### 7. **迭代器**
1. `begin` 和 `end`

用于遍历容器。

```cpp
for (auto it = map1.begin(); it != map1.end(); ++it) {
    std::cout << it->first << " -> " << it->second << std::endl;
}
```

2. 范围 for 循环

```cpp
for (const auto& pair : map1) {
    std::cout << pair.first << " -> " << pair.second << std::endl;
}
```
