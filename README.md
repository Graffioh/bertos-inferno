# What's berto's inferno?

the main purpose of this, is to explain and internalize these weird solutions that i could never come up with on my own, with the goal of preparing for interviews and pushing through this hell up to heaven, like dante with beatrice

![bertosinferno-img](./img/bertosinferno.jpg)

# What's my current approach?

i'm randomly working through problems from [sean prashad list](https://seanprashad.com/leetcode-patterns/) and i'm also considering integrating anki for spaced repetition (i'm doing it 'by hand' rn)

## [46. Permutations](https://leetcode.com/problems/permutations/description/) | [77. Combinations](https://leetcode.com/problems/combinations/description/) | [78. Subsets](https://leetcode.com/problems/subsets/description/)

### key idea

these 3 problems are quite similar, the purpose is to arrange the given numbers based on some constraints

tipically when we want to arrange something, we use *backtracking* because recursion provides us an handy way to handle arrangements

### Permutations

**constraints**
- literally every elements but in different order

~~~py
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        permutations = []
        checked = [False] * 21

        def dfs(i):
            if i == len(nums):
                res.append(permutations[:])
                return
            
            for j in range(len(nums)):
                if checked[j]:
                    continue
                
                checked[j] = True
                permutations.append(nums[j])
                
                dfs(i + 1)

                checked[j] = False
                permutations.pop()

        dfs(0)
        return res
~~~

**complexity**
~~~
time = O(N! * N^2) 
~~~
n! = order of permutations & n*n = n elements insertion into permutations array for each iterations

~~~
space = O(N! * N) 
~~~
if we are not counting the res array, because each permutation is of size n! and we building permutations iteratively so n


### Combinations

**constraints**
- each combination of numbers n must be of size k 
- no repetitions

~~~py
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        combinations = []

        def dfs(i):
            if i > n + 1:
                return

            if len(combinations) == k:
                res.append(combinations[:])
                return

            for j in range(i, n + 1):
                combinations.append(j)
                dfs(j + 1)
                combinations.pop()

        dfs(1)
        return res
~~~

**complexity**
~~~
time = O(N choose K * K) 
~~~
the dfs generates all combinations of k elements from a set of n elements (binomial coefficient)
for each combination, there is an appending operation of k-elements to res array

~~~
space = O(N choose K) 
~~~
n choose k = order of combinations


### Subsets

**constraints**
- can contain empty array
- order is important

~~~py
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = [] 
        subset = []

        def dfs(i):
            if i == len(nums):
                res.append(subset[:])
                return

            subset.append(nums[i])
            dfs(i + 1)
            subset.pop()
            dfs(i + 1)
    
        dfs(0)
        return res
~~~

**complexity**
~~~
time = O(N * 2^N) 
~~~
n = how many subsets do we need & 2^n = subset order

~~~
space = O(2^N) 
~~~
2^N = order of subsets


## [238. Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/description/)

### key idea
- compute the prefix product (the product of all the elements preceding the current element)
- computer the suffix product (the product of all the elements following the current element)
- the product of these two arrays results in an array where each element is the product of all elements except itself

~~~py
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        prefix = [1] * (len(nums) + 1)
        suffix = [1] * (len(nums) + 1)
        res = [0] * len(nums)

        for i in range(len(nums)):
            prefix[i + 1] = prefix[i] * nums[i]

        for i in range(len(nums) - 1, 0, -1):
            suffix[i - 1] = suffix[i] * nums[i]

        for i in range(len(nums)):
            res[i] = prefix[i] * suffix[i] 

        return res
~~~

**complexity**
~~~
time = O(N) 
~~~
since we compute prefix, suffix and res by only going through all the elements once each iteration

~~~
space = O(N) 
~~~
since the auxiliary arrays (prefix and suffix) are of size n + 1

### optimization

the solution above could be simplified by 'merging' together the prefix and suffix arrays, so by doing the operation in place, but focus on the first solution for a true understanding of the problem since the optimized one is less obvious

~~~py
class Solution:
    # example: [1, 2, 3, 4]
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        res = [1] * len(nums)

        prefix = 1
        for i in range(len(nums)):
            res[i] = prefix
            prefix = prefix * nums[i]

        # print(res) -> [1, 1, 2, 6]

        suffix = 1
        for i in range(len(nums) - 1, -1, -1):
            # [6 * 1, 2 * 4, 1 * 12, 1 * 24] in reverse since
            #    the loop is decrementing
            res[i] = res[i] * suffix

            # 1 -> 4 -> 12 -> 24 -> 24
            suffix = suffix * nums[i]

        # print(res) -> [24, 12, 8, 6] 

        return res
~~~

**complexity**
~~~
time = O(N) 
~~~
like before

~~~
space = O(1) 
~~~
since there are no auxiliary arrays except the result array

### useful resources

[prefix sum video by errichto](https://www.youtube.com/watch?v=bNvIQI2wAjk)

## 314. Binary Tree Vertical Order Traversal [(premium)](https://leetcode.com/problems/binary-tree-vertical-order-traversal/description) | ['premium'](https://www.lintcode.com/problem/651/)

### key idea
each node has a column associated to it and based on the columns the node values must be put in an hashmap structured this way column (key) : node (value)

there is a simple calculation to do with columns

- go left -> column - 1
- go right -> column + 1

the hashmap must be populated using **bfs** and then sorted otherwise with dfs you wont have the correct order

~~~py
class Solution:
    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        res = []
        hmap = defaultdict(list)
        queue = deque([(0, root)])

        while queue:
            col, node = queue.popleft()

            hmap[col].append(node.val)

            if node.left:
                queue.append((col - 1, node.left))
            if node.right:
                queue.append((col + 1, node.right))
        
        sortHmap = dict(sorted(hmap.items()))

        for item in sortHmap.values():
            res.append(item)

        return res
~~~

**optimization**

this little optimization eliminates the need to sort the hashmap, basically by defining a range of columns from *min_col* to *max_col*

~~~py
class Solution:
    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        res = []
        hmap = defaultdict(list)
        queue = deque([(0, root)])
        
        min_col = float(inf)
        max_col = float(-inf)

        while queue:
            col, node = queue.popleft()

            hmap[col].append(node.val)

            min_col = min(min_col, col)
            max_col = max(max_col, col)

            if node.left:
                queue.append((col - 1, node.left))
            if node.right:
                queue.append((col + 1, node.right))
        
        for col in range(min_col, max_col + 1):
            res.append(hmap[col])

        return res
~~~

**complexity**
~~~
time = O(N) 
~~~
n = height of the tree in the worst case

~~~
space = O(N) 
~~~
because we are only storing values of the tree in the hashmap/result and there are n values


## [543. Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree)

### key idea
the thing to remember for this problem that i always forget, is that the diameter could be:

- diameter = current node left path + current node right path
- diameter = curret node left OR right path + old left OR right path

that's why we need a nonlocal variable, since the diameter for some cases cannot be calculated solely from the current node but requires information from previous dfs calls as well

~~~py
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        diameter = 0

        def dfs(node) -> int:
            if not node:
                return 0
            nonlocal diameter

            leftPath = dfs(node.left)
            rightPath = dfs(node.right) 

            diameter = max(diameter, leftPath + rightPath)
            
            return max(leftPath, rightPath) + 1
            
        dfs(root)
        return diameter
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through each node once with the dfs

~~~
space = O(N) 
~~~
the call stack is based on the height of the tree and in the worst case it could be O(N)
