---
date: 2025-07-22T11:00:59-04:00
description: "JSON（JavaScript Object Notation）是一种轻量级的数据交换格式，采用完全独立于语言的文本格式来存储和传输数据。"
featured_image: "/images/json/jaz.png"
tags: ["data"]
title: "JSON"
---

## JSON Types

+ **Strings**："Hello World" "Kyle"

+ **Numbers**：10 1.5 -30 1.2e10

+ **Booleans**：true false
+ **null**：null
+ **Arrays**：[1, 2, 3] ["Hello", "World"]
+ **Objects**：{ "key": "value" }: {"age": 30}

## 可嵌套

```json
{
	"name": "Kyle"
	"favoriteNumber": 3,
	"isProgrammer": true,
	"hobbies": ["Weight Lifting",
	"Bowling"],
	"friends": [{
		"name": "Joey",
		"favoriteNumber": 100,
		"isProgrammer": false,
		"friends": [...]
	}]
}
```

&nbsp;

<!--more-->

## 实例 (companies.json)

```json
[
	{
		"name": "Big Corporation",
		"numberOfEmployees": 10000,
		"ceo": "Mary"
		"rating": 3.6
	},
	{
		"name": "Small Startup",
		"numberOfEmployees": 3,
		"ceo": null
	}
]
```

