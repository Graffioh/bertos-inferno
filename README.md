# What's berto's inferno?

the main purpose of this, is to explain and internalize these weird solutions that i could never come up with on my own, with the goal of preparing for interviews and pushing through this hell up to heaven, like dante with beatrice

![bertosinferno-img](./img/bertosinferno.jpg)

# What's my current approach?

anki deck for spaced repetition and

- **leetcode free:**
randomly working through problems from [sean prashad list](https://seanprashad.com/leetcode-patterns/)

- **leetcode premium:**
following company specific problems based on frequency

i try solving the question for 30 minutes, then look at the solution until i fully understand it

# Anki

- [download deck](https://ankiweb.net/shared/info/1983665439?cb=1725474892013)

# Resources

- [neetcode](https://www.youtube.com/c/neetcode)
- [cracking faang](https://www.youtube.com/@crackfaang)
- [code with carter](https://www.youtube.com/@codewithcarter)
- [competitive programming handbook](https://cses.fi/book/book.pdf)
- [dynamic programming book](https://dp-book.com/Dynamic_Programming.pdf)
- [snats xeet](https://x.com/snats_xyz/status/1832178008578224551)

# Problems index
- [3. Longest Substring Without Repeating Characters](#3-longest-substring-without-repeating-characters)
- [5. Longest Palindromic Substring](#5-longest-palindromic-substring)
- [7. Reverse Integer](#7-reverse-integer)
- [14. Longest Common Prefix](#14-longest-common-prefix)
- [15. 3Sum](#15-3sum)
- [17. Letter Combinations of a Phone Number](#17-letter-combinations-of-a-phone-number)
- [23. Merge k Sorted Lists](#23-merge-k-sorted-lists)
- [31. Next Permutation](#31-next-permutation)
- [33. Search in Rotated Sorted Array](#33-search-in-rotated-sorted-array)
- [34. Find First and Last Position of Element in Sorted Array](#34-find-first-and-last-position-of-element-in-sorted-array)
- [36. Valid Sudoku]()
- [46. Permutations | 77. Combinations | 78. Subsets](#46-permutations--77-combinations--78-subsets)
- [49. Group Anagrams](#49-group-anagrams)
- [50. Pow(x,n)](#50-powxn)
- [54. Spiral Matrix](#54-spiral-matrix)
- [56. Merge Intervals](#56-merge-intervals)
- [71. Simplify Path](#71-simplify-path)
- [88. Merge Sorted Array](#88-merge-sorted-array)
- [128. Longest Consecutive Sequence](#128-longest-consecutive-sequence)
- [129. Sum Root to Leaf Numbers](#129-sum-root-to-leaf-numbers)
- [133. Clone Graph](#133-clone-graph)
- [138. Copy List with Random Pointer](#138-copy-list-with-random-pointer)
- [146. LRU Cache](#146-lru-cache)
- [162. Find Peak Element](#162-find-peak-element)
- [199. Binary Tree Right Side View](#199-binary-tree-right-side-view)
- [207. Course Schedule](#207-course-schedule)
- [210. Course Schedule II](#210-course-schedule-ii)
- [215. Kth Largest Element in an Array](#215-kth-largest-element-in-an-array)
- [227. Basic Calculator II](#227-basic-calculator-ii)
- [236. Lowest Common Ancestor of a Binary Tree](#236-lowest-common-ancestor-of-a-binary-tree)
- [238. Product of Array Except Self](#238-product-of-array-except-self)
- [314. Binary Tree Vertical Order Traversal](#314-binary-tree-vertical-order-traversal-premium--premium)
- [339. Nested List Weight Sum](#339-nested-list-weight-sum-premium)
- [346. Moving Average from Data Stream](#346-moving-average-from-data-stream-premium--premium) 
- [347. Top K Frequent Elements](#347-top-k-frequent-elements)
- [398. Random Pick Index](#398-random-pick-index)
- [415. Add Strings](#415-add-strings)
- [426. Convert Binary Search Tree to Sorted Doubly Linked List](#426-convert-binary-search-tree-to-sorted-doubly-linked-list-premium--premium) 
- [498. Diagonal Traverse](#498-diagonal-traverse)
- [523. Continuous Subarray Sum](#523-continuous-subarray-sum)
- [525. Contiguous Array](#525-contiguous-array)
- [528. Random Pick with Weight](#528-random-pick-with-weight)
- [543. Diameter of Binary Tree](#543-diameter-of-binary-tree)
- [560. Subarray Sum Equals K](#560-subarray-sum-equals-k)
- [636. Exclusive Time of Functions](#636-exclusive-time-of-functions)
- [637. Valid Word Abbreviation](#637-valid-word-abbreviation-premium--premium)
- [647. Palindromic Substrings](#647-palindromic-substrings)
- [670. Maximum Swap](#670-maximum-swap)
- [680. Valid Palindrome II](#680-valid-palindrome-ii)
- [708. Insert into a Sorted Circular Linked List](#708-insert-into-a-sorted-circular-linked-list)
- [721. Accounts Merge](#721-accounts-merge)
- [767. Reorganize String](#767-reorganize-string)
- [791. Custom Sort String](#791-custom-sort-string)
- [827. Making A Large Island](#827-making-a-large-island)
- [875. Koko Eating Bananas](#875-koko-eating-bananas)
- [921. Minimum Add to Make Parentheses Valid](#921-minimum-add-to-make-parentheses-valid)
- [938. Range Sum of BST](#938-range-sum-of-bst)
- [953. Verifying an Alien Dictionary](#953-verifying-an-alien-dictionary)
- [973. K Closest Points to Origin](#973-k-closest-points-to-origin)
- [986. Interval List Intersections](#986-interval-list-intersections)
- [994. Rotting Oranges](#994-rotting-oranges)
- [1004. Max Consecutive Ones III](#1004-max-consecutive-ones-iii)
- [1091. Shortest Path in Binary Matrix](#1091-shortest-path-in-binary-matrix)
- [1249. Minimum Remove to Make Valid Parentheses](#1249-minimum-remove-to-make-valid-parentheses)
- [1539. Kth Missing Positive Number](#1539-kth-missing-positive-number)
- [1570. Dot Product of Two Sparse Vectors](#1570-dot-product-of-two-sparse-vectors-premium--premium)
- [1650. Lowest Common Ancestor of a Binary Tree III](#1650-lowest-common-ancestor-of-a-binary-tree-iii-premium)
- [1762. Buildings With an Ocean View](#1762-buildings-with-an-ocean-view-premium)
- [1868. Product of Two Run Length Encoded Arrays](#1868-product-of-two-run-length-encoded-arrays-premium--premium) 
- [2055. Plates Between Candles](#2055-plates-between-candles)
- [2340. Minimum Adjacent Swaps to Make a Valid Array](#2340-minimum-adjacent-swaps-to-make-a-valid-array-premium)

---

## [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters)

### key idea

sliding window, store indexes inside a dictionary, as soon as we encounter a character that is in the dictionary we need to check if it's in the range and update the left pointer

~~~py
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s) <= 1:
            return len(s)
            
        char_dict_idx = defaultdict(int)
        res = 0
        l = 0

        for r in range(len(s)):
            if s[r] in char_dict_idx:
                if char_dict_idx[s[r]] >= l:
                    l = char_dict_idx[s[r]] + 1

            res = max(res, r - l + 1)
            char_dict_idx[s[r]] = r
        
        return res
~~~

**complexity**
~~~
time = O(N)
~~~
we go through the whole string only once

~~~
space = O(N)
~~~
dictionary size, n size of the string

## [5. Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring)

### key idea

each letter is a different center, from the center we expand to the right and to the left to check the palindrome

check first odd lengt then even length palindrome

for each iteration, calculate the max length

~~~py
class Solution:
    def longestPalindrome(self, s: str) -> str:
        res = ""
        length = 0

        for i in range(len(s)):
            # odd length palindrome
            l, r = i, i
            while l >= 0 and r < len(s) and s[l] == s[r]:
                if (r - l + 1) > length:
                    res = s[l:r+1]
                    length = r - l + 1
                l -= 1
                r += 1

            # even length palindrome
            l, r = i, i + 1
            while l >= 0 and r < len(s) and s[l] == s[r]:
                if (r - l + 1) > length:
                    res = s[l:r+1]
                    length = r - l + 1
                l -= 1
                r += 1
        
        return res
~~~

**complexity**
~~~
time = O(N)
~~~
process the whole string

~~~
space = O(1)
~~~
no extra space used

## [7. Reverse Integer](https://leetcode.com/problems/reverse-integer)

### key idea

use % (to pick the last digit) and / (to remove the last digit)

by picking the last digit each iteration, you can add that to the result and this will give us the reversed integer

each iteration should remove one digit so when no more digits are left, the loop ends

the two ifs are used to check for overflows

~~~py
class Solution:
    def reverse(self, x: int) -> int:
        MAX = 2147483647
        MIN = -2147483648
        res = 0

        while x:
            digit = int(math.fmod(x, 10))
            x = int(x / 10)

            if res > MAX // 10 or res == MAX // 10 and digit >= MAX % 10:
                return 0

            if res < MIN // 10 or res == MIN // 10 and digit <= MIN % 10:
                return 0
            
            res = (res * 10) + digit
        
        return res
~~~

**complexity**
~~~
time = O(N)
~~~
process the whole number

~~~
space = O(1)
~~~
no extra space used


## [14. Longest Common Prefix](https://leetcode.com/problems/longest-common-prefix)

### key idea

vertical scanning

pick a base, loop through all the words and as soon as one character is different from the base or we went through the whole word (this will hit when we encounter the shortest word), then return the prefix from the start to the point we arrived 

~~~py
class Solution:
    def longestCommonPrefix(self, v: List[str]) -> str:
        if len(v) == 0:
            return ""

        base = v[0]
        for i in range(len(base)):
            for word in v[1:]:
                if i == len(word) or word[i] != base[i]:
                    return base[0:i]
        
        return base
~~~

**complexity**
~~~
time = O(S)
~~~
S -> the sum of all chars in all strings, in the worst case n equal strings with length m so S = n * m

~~~
space = O(1)
~~~
no extra space used

## [15. 3Sum](https://leetcode.com/problems/3sum/description)

### key idea

use three pointers, one fixed at the beginning, two that moves from i + 1 to nums.length

the key here is:

- check if the total is < 0, > 0 or == 0
- < 0, increment j so the number gets bigger
- \> 0, decrement k so the number gets smaller
- == 0 append to res and move the two pointers
- <ins>skip whenever i or j or k number is the same as the one before</ins>

~~~py
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()

        for i in range(len(nums) - 2):
            # skip i duplicates
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            j, k = i + 1, len(nums) - 1
            while j < k:
                if nums[i] + nums[j] + nums[k] < 0:
                    j += 1
                elif nums[i] + nums[j] + nums[k] > 0:
                    k -= 1
                else:
                    res.append([nums[i], nums[j], nums[k]])
                    j += 1
                    k -= 1

                    # skip j duplicates
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1

                    # skip k duplicates
                    while j < k and nums[k] == nums[k + 1]:
                        k -= 1
        
        return res
~~~

**complexity**
~~~
time = O(N^2 + NlogN) = O(N^2)
          ^       ^
    for & while  sorting
~~~

~~~
space = O(1) or O(N) <--
~~~
in python the sort takes N of space complexity due to how the library implemented the method

## [17. Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number)

### key idea

hashmap to map the numbers to its respective letters

then backtrack to explore all different possibilities going by each digit one by one

~~~py
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []

        num_to_chars = {
            "2": ["a", "b", "c"], 
            "3": ["d", "e", "f"], 
            "4": ["g", "h", "i"],
            "5": ["j", "k", "l"], 
            "6": ["m", "n", "o"], 
            "7": ["p", "q", "r", "s"], 
            "8": ["t", "u", "v"], 
            "9": ["w", "x", "y", "z"]
            }
        res = []

        def dfs(i, cur_string):
            if i == len(digits):
                res.append("".join(cur_string))
                return

            for c in num_to_chars[digits[i]]:
                cur_string.append(c)
                dfs(i + 1, cur_string)
                cur_string.pop()
        
        dfs(0, [])
        return res
~~~

**complexity**
~~~
time = O(4^N * N)
~~~
in the worst case we could get N digits of only '7' and/or '9' and as you can see its corresponding letters array is of length 4

~~~
space = O(N) or O(4^N) if we count recursive stack space
~~~

## [23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists)

### key idea

heap solution works but it's not optimized since the heap take some space

the most optimal solution is merging the lists two by two by defining an interval where we pick the two lists

each time this interval doubles so the result halves each time and at the end we are going to have the first list that represents the result, so all the lists merged into the first one

~~~py
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if not lists:
            return None
            
        def merge(l1, l2):
            if not l1:
                return l2
            if not l2:
                return l1
            else:
                if l1.val <= l2.val:
                    l1.next = merge(l1.next, l2)
                    return l1
                else:
                    l2.next = merge(l1, l2.next)
                    return l2

        interval = 1

        while interval < len(lists):
            for i in range(0, len(lists) - interval, interval * 2):
                lists[i] = merge(lists[i], lists[i + interval])
            interval *= 2
        return lists[0]
~~~

**complexity**
~~~
time = O(N*logK)
~~~
N -> total number of elements\\
log(k) -> we halves each time the input by picking only two lists at a time

~~~
space = O(1)
~~~
no extra memory used, the result will be the first list that is already present in the input

## [31. Next Permutation](https://leetcode.com/problems/next-permutation)

### key idea

the idea is simple, why it works is a little bit harder to explain

<ins>**idea**</ins>
since it must be done in place, there needs to be some sort of swapping

if we look from right to left and all the numbers are in increasing order, then there is no next lexicographical permutation

now we must go in reverse order, finding the pivot, then after the pivot is found we must swap the pivot with the first number that is greater than the pivot from right to left

and after this reverse the part of the array after the pivot

<ins>**why it works?**</ins>

going from right to left and finding the pivot, means that at the right of the pivot, since the array is in increasing order (reversed), there can't be some rearrangement that create a larger lexicographical permutation

now we want to create a permutation that is just larger than the current one, therefore we need to replace the pivot with the number which is just larger than itself among the numbers on the right part of the pivot

something like this lol

~~~py
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        # 1 4 5 8 7

        pivot = None

        # find the pivot
        for i in range(len(nums) - 1, 0, -1):
            if nums[i] > nums[i - 1]:
                pivot = i - 1
                break
        else: # if not found (full iteration done), then just return the 
              #     reversed array as in the example
            nums.reverse()
            return
        
        # swap the pivot with the first greater num found from right to left
        swap = len(nums) - 1
        while nums[swap] <= nums[pivot]:
            swap -= 1

        nums[swap], nums[pivot] = nums[pivot], nums[swap]

        # reverse the part after the pivot because yes
        nums[pivot + 1:] = reversed(nums[pivot + 1:])
~~~

**complexity**
~~~
time = O(N + N) -> O(N)
~~~
in the worst case the for loop finishes, so full iteration + reversing

~~~
space = O(1)
~~~
all done in-place, baby :*

## [33. Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array)

### key idea

binary search ofc

since we know that the array was sorted before, that means that there should be a pivot that splits the array in two sorted parts

play around that pivot and find the target

~~~py
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1

        while l <= r:
            mid = (l + r) // 2

            if nums[mid] == target:
                return mid
            elif nums[mid] >= nums[l]:
                if nums[l] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                if nums[mid] < target <= nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1
        return -1
~~~

**complexity**
~~~
time = O(logN)
~~~
binary search

~~~
space = O(1)
~~~
all done in-place

## [34. Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description)

### key idea

it's a binary search slightly modified

basically we search for the left most and right most index of the target value, using two binary search

~~~py
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def search_idx(is_left_check):
            l, r = 0, len(nums) - 1
            idx = -1

            while l <= r:
                mid = (l + r) // 2

                if nums[mid] < target:
                    l = mid + 1
                elif nums[mid] > target:
                    r = mid - 1
                else:
                    idx = mid
                    if is_left_check:
                        r = mid - 1
                    else:
                        l = mid + 1
            return idx

        left_idx = search_idx(True)
        right_idx = search_idx(False)

        return [left_idx, right_idx]
~~~

**complexity**
~~~
time = O(logN)
~~~
binary search, input halves each time

~~~
space = O(1)
~~~
no extra memory used

## [36. Valid Sudoku](https://leetcode.com/problems/valid-sudoku/description/)

### key idea

use different sets for each row, col and grid

to differentiate the grid, we can assume that each grid is one position, so top left is 0, top center is 1, top right is 2 and so on

with this differentiation we can index inside the grid dictionary simply dividing by 3

~~~py
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        row_set = defaultdict(set)
        col_set = defaultdict(set)
        box_set = defaultdict(set)

        for i in range(len(board)):
            for j in range(len(board[0])):
                val = board[i][j]
                if val == ".":
                    continue
                    
                if val in row_set[i] or val in col_set[j] or val in box_set[(i // 3, j // 3)]:
                    return False

                row_set[i].add(val)
                col_set[j].add(val)
                box_set[(i // 3, j // 3)].add(val)
        
        return True
~~~

**complexity**
~~~
time = O(N*M) = O(9x9) = O(1)
~~~
we go through the whole board in the worst case

~~~
space = O(9x9)
~~~
same as time for the dictionaries

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

## [49. Group Anagrams](https://leetcode.com/problems/group-anagrams)

### key idea

hashmap that map the count of characters with the relative word that have the same count

we count character using an array that we later transform in a tuple to group the words together

~~~py
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        count_dict = defaultdict(list) # char count -> list of anagrams

        for word in strs:
            count = [0] * 26

            for c in word:
                count[ord(c) - ord("a")] += 1
            
            count_dict[tuple(count)].append(word)
        
        return count_dict.values()
~~~

**complexity**
~~~
time = O(MN) 
~~~
for each string we iterate each character

~~~
space = O(N) 
~~~
hashmap size

## [50. Pow(x,n)](https://leetcode.com/problems/powx-n/description)

### key idea
the iterative approach of multiplying n times is not optimized

so we need to find a way to compute half of the calculations

the basic idea is to divide up exponents: 
- 2^4 = 2^2 * 2^2 = 2^1 * 2^1 * 2^1 * 2^1
- 2^5 = 2^1 * 2^4 = ...

if it's even, then divide by 2, if it's odd we should 'remove' 1 and then divide by 2 the remaining part

thanks to the halving, we can memoize the computations efficiently

~~~py
class Solution:
    def myPow(self, x: float, n: int) -> float:
        isExpNegative = n < 0
        n = abs(n)
        expDict = defaultdict(int)

        def dfs(exp) -> float:
            if exp == 0:
                return 1
            
            if exp == 1:
                return x
            
            if exp in expDict:
                return expDict[exp]
            
            # (x if exp % 2 == 1 else 1) is for dividing up odd number 
            expDict[exp] = dfs(exp // 2) * dfs(exp // 2) * (x if exp % 2 == 1 else 1)

            return expDict[exp]
        
        return 1/dfs(n) if isExpNegative else dfs(n)
~~~

**complexity**
~~~
time = O(logN) 
~~~
we cut by 2 the computation eachtime

~~~
space = O(logN) 
~~~
the dictionary stores only 'intermediate' results thanks to the halving

## [54. Spiral Matrix](https://leetcode.com/problems/spiral-matrix)

### key idea

we gonna use left, right, top, bottom pointers to mark the boundaries

for each iteration we update those boundaries and make the matrix "smaller"

bullshit

~~~py
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        l, r = 0, len(matrix[0])
        t, b = 0, len(matrix)
        res = []

        while l < r and t < b:
            # left to right
            for i in range(l, r):
                res.append(matrix[t][i])
            t += 1

            # top to bottom
            for i in range(t, b):
                res.append(matrix[i][r - 1])
            r -= 1

            # check if right went before left or top went after bottom (since we update them above)
            if not (l < r and t < b):
                break
            
            # right to left
            for i in range(r - 1, l - 1, -1):
                res.append(matrix[b - 1][i])
            b -= 1

            # bottom to top
            for i in range(b - 1, t - 1, -1):
                res.append(matrix[i][l])
            l += 1
        return res
~~~

**complexity**
~~~
time = O(N * M) 
~~~
we go through the whole matrix

~~~
space = O(1) or O(N * M) if we count res
~~~

## [56. Merge Intervals](https://leetcode.com/problems/merge-intervals)

## key idea
first sort the intervals by the first value and put the first interval inside res

res is gonna manage the merged intervals (and it's gonna be the final result obv)

when to merge an interval? if the previous interval END is grater or equal than the current interval START, really simple

~~~py
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if len(intervals) == 1:
            return intervals
        
        intervals.sort()

        res = [intervals[0]]

        for start, end in intervals:
            if res[-1][1] >= start:
                res[-1][1] = max(res[-1][1], end)
            else:
                res.append([start, end])
            
        return res
~~~

**complexity**
~~~
time = O(N) 
~~~
we iterate through all the intervals

~~~
space = O(N) 
~~~
in the worst case all the intervals are disjoint (no overlap) so |res| = |intervals|

## [71. Simplify Path](https://leetcode.com/problems/simplify-path)

### key idea
the only relevant special character for this problem is ".."

use a stack for storing words, whenever ".." is encountered if there are words in the stack, pop

for "" and "." just continue the iteration cause we just skip them

at the end we join all the words from the stack by separating them with "/"

~~~py
class Solution:
    def simplifyPath(self, path: str) -> str:
        words_stk = []
        items = path.split("/")

        for item in items:
            if item == "" or item == ".":
                continue
            
            if item == "..":
                if words_stk:
                    words_stk.pop()
            else:
                words_stk.append(item)
        
        return "/" + "/".join(words_stk)
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through all the path of size n

~~~
space = O(N) 
~~~
we store words from the path in the stack

## [88. Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array)

## key idea
use 3 pointers, that points:

- at the last position of nums1 (last)
- at the last element of nums1 (p1)
- at the last element of nums2 (p2)

now it consists of comparing p1 and p2, whoever is the largest, just put it in the last position and decrement the pointers accordingly

there is one edge case where p1 becomes 0 but there are still p2 elements remaining in nums2, in that case pour all of them in nums1

~~~py
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        last = m + n - 1
        p1 = m - 1
        p2 = n - 1

        while p1 >= 0 and p2 >= 0:
            if nums1[p1] < nums2[p2]:
                nums1[last] = nums2[p2]
                p2 -= 1
            else:
                nums1[last] = nums1[p1]
                p1 -= 1
            last -= 1
        
        # edge case: if there are remaining elements in nums2
        while p2 >= 0:
            nums1[last] = nums2[p2]
            p2 -= 1
            last -= 1
~~~

**complexity**
~~~
time = O(N + M) 
~~~
we go through both of the arrays once

~~~
space = O(1) 
~~~
we do the operations in place without extra memory

## [128. Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/description/)

### key idea

we need to careful decide where to start searching the sequence

the right starting point is a number without any preceding number

for example: [100,4,200,1,3,2] the starting point would be 1 and from 1 we iterate on the right to search for its consecutives (based on a set, to remove duplicates)

~~~py
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        nums_set = set(nums)
        max_streak = 0

        for n in nums:
            if n - 1 not in nums_set:
                cur = n
                cur_streak = 1

                while cur + 1 in nums_set:
                    cur_streak += 1
                    cur += 1

                max_streak = max(max_streak, cur_streak)
        
        return max_streak
~~~

**complexity**
~~~
time = O(N) 
~~~
set creation and iteration through the whole array nums

~~~
space = O(N) 
~~~
set memory

## [129. Sum Root to Leaf Numbers](https://leetcode.com/problems/sum-root-to-leaf-numbers)

### key idea

really simple just read the code

~~~py
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        def dfs(node, num):
            if not node:
                return 0
            
            num = num * 10 + node.val
            if not node.left and not node.right:
                return num
            
            return dfs(node.left, num) + dfs(node.right, num)
        
        return dfs(root, 0)
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through the whole tree

~~~
space = O(1) or O(height) if recursive stack is counted
~~~

## [133. Clone Graph](https://leetcode.com/problems/clone-graph)

### key idea

use an hashmap to keep the old_node : cloned_node mapping

do a bfs, clone the nodes and remember to populate neighbors array for each clone

~~~py
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if not node:
            return None

        cloned = {}
        cloned[node] = Node(node.val, [])

        q = deque([node])

        while q:
            cur = q.popleft()

            for neigh in cur.neighbors:
                if neigh not in cloned:
                    cloned[neigh] = Node(neigh.val, [])
                    q.append(neigh)
                
                cloned[cur].neighbors.append(cloned[neigh])

        return cloned[node]
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through the whole graph

~~~
space = O(N) 
~~~
hashmap size, storing all the nodes of the graph

## [138. Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer)

### key idea

there is a simple way that involves an hash map but that would take O(n) space

this approac ensures O(1) space thanks to the interleaving of the nodes

basically we create a specular copy / clone of each node as their next

then we assign the random pointers to the cloned nodes thanks to the original ones

at the end we skip the original nodes by skipping the original ones, and thanks to this trick we'll have a deep copy of the original list due to different references for each node

~~~py
class Solution:
  def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head: 
            return head

        # create cloned node as the next of the respective original node
        cur = head
        while cur:
            newnode = Node(cur.val, cur.next)
            cur.next = newnode
            cur = newnode.next
        
        # assign the random original node pointer to the cloned node
        cur = head
        while cur:
            cur.next.random = cur.random.next if cur.random else None
            cur = cur.next.next
        
        # skip original nodes to return only the cloned nodes
        cur = head.next
        while :
            cur.next = cur.next.next if cur.next else None
            cur = cur.next
        
        return head.next
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through the whole list

~~~
space = O(1) 
~~~
no extra space used thanks to interleaving nodes


## [146. LRU Cache](https://leetcode.com/problems/lru-cache/description)

### key idea

the intuition is to have an hash map that stores these key value pairs

the value is not actually the value but is a pointer to a node that stores key and value

thanks to this we can use pointers to handle the lru mechanism

double linked list

left and right pointers (left = LRU, right = MRU)

watch neetcode video in resources, his explanation is really good

~~~py
class Node:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.next = self.prev = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.kv = {}
        self.left, self.right = Node(0,0), Node(0,0)
        self.left.next, self.right.prev = self.right, self.left
    
    # insrt to right (MRU, Most Recently Used)
    def insert(self, node):
        prev, nxt = self.right.prev, self.right
        prev.next = node
        nxt.prev = node
        node.next = nxt
        node.prev = prev
    
    def remove(self, node):
        prev, nxt = node.prev, node.next
        prev.next = nxt
        nxt.prev = prev

    def get(self, key: int) -> int:
        if key in self.kv:
            self.remove(self.kv[key])
            self.insert(self.kv[key])
            return self.kv[key].val
        return -1

    def put(self, key: int, value: int) -> None:
        # to update if it already exists (put it on most right position)
        if key in self.kv:
            self.remove(self.kv[key])
        
        self.kv[key] = Node(key, value)
        self.insert(self.kv[key])

        # eviction of LRU
        if len(self.kv) > self.cap:
            lru = self.left.next
            self.remove(lru)
            del self.kv[lru.key]
~~~

**complexity**

~~~
time = O(1) 
~~~
no iterations or expensive operations

~~~
space = O(N) 
~~~
the hash map

### resources

- [neetcode video](https://www.youtube.com/watch?v=7ABFKPK2hD4)


## [162. Find Peak Element](https://leetcode.com/problems/find-peak-element)

### key idea
just compare the mid value, literally a binary search with some alterations

remember to put the right/left to -inf if needed (mid - 1 or mid + 1 out of bound)

~~~py
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return 0
        
        if len(nums) == 2:
            return 0 if nums[0] > nums[1] else 1
        
        l,r = 0, len(nums) - 1
        
        while l <= r:
            mid = (l + r) // 2

            mid_val = nums[mid]
            right_val = nums[mid + 1] if mid < len(nums) - 1 else float(-inf)
            left_val = nums[mid - 1] if mid > 0 else float(-inf)

            if left_val < mid_val > right_val:
                return mid
            elif mid_val < right_val:
                l = mid + 1
            else:
                r = mid - 1
        
        return 0
~~~

**complexity**
~~~
time = O(logN) 
~~~
as the problem statement requested, we half it each time

~~~
space = O(1) 
~~~
no extra space is used

## [199. Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view/description)

### key idea

really simple, since you need to return the right most value, we leverage the concept of the BFS to process the whole level and as soon as we arrive at the last element of the processed level, then we append it to the res

~~~py
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []

        queue = deque([root])
        res = []

        while queue:
            cur_level_length = len(queue)

            for i in range(cur_level_length):
                node = queue.popleft()

                if i == cur_level_length - 1:
                    res.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        
        return res
~~~

**complexity**
~~~
time = O(N) 
~~~
since we touch every single node of the tree due to bfs 

~~~
space = O(N) 
~~~
it depends on the height of the tree, but in the worst case (complete tree) we gonna store n elements

## [207. Course Schedule](https://leetcode.com/problems/course-schedule/)

### key idea

here if you do some examples, you can see that:
- this is a graph problem
- you just need to check for cycles

so construct a graph, do a dfs to check for cycles and that's it

~~~py
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        graph = defaultdict(list)

        # 0 = white, 1 = grey, 2 = black
        visited = [0] * (numCourses)

        # graph construction
        for src, dst in prerequisites:
            graph[src].append(dst)

        # dfs to check for cycles
        def courseCheck(node, check):
            visited[node] = 1

            for adj in graph[node]:
                if visited[adj] == 1:
                    return False

                if visited[adj] == 0:
                    check = courseCheck(adj, check)
                    if check == False:
                        return False
                
            visited[node] = 2
            return check
        
        # dfs every source
        vertices = list(graph.keys())
        for v in vertices:
            if visited[v] == 0:
                check = courseCheck(v, True)
                if check == False:
                    return False

        return True
~~~

**complexity**
~~~
time = O(V + E) 
~~~
typical graph dfs complexity, we go through all edges and pass through all vertices in the worst case -> we traverse the whole graph

~~~
space = O(V) 
~~~
graph size, number of vertices

## [210. Course Schedule II](https://leetcode.com/problems/course-schedule-ii)

### key idea

same as course schedule, but here we must return a list based on the path of dfs visit

if there is a loop, return an empty list

~~~py
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        res = []
        graph = {i: [] for i in range(numCourses)}
        visited = [0] * numCourses

        for src, dst in prerequisites:
            graph[src].append(dst)
        
        def dfs(node, check):
            visited[node] = 1

            for adj in graph[node]:
                if visited[adj] == 1:
                    return True

                if visited[adj] == 0:
                    check = dfs(adj, check)
                    if check == True:
                        return True
            
            if visited[node] != 2:
                res.append(node)
            visited[node] = 2
        
        for src in list(graph):
            if visited[src] == 0:
                check = dfs(src, False)
                if check == True:
                    return []
        
        return res
~~~

**complexity**
~~~
time = O(V + E) 
~~~
typical graph dfs complexity, we go through all edges and pass through all vertices in the worst case -> we traverse the whole graph

~~~
space = O(V) 
~~~
graph size, number of vertices

## [215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/description)

### key idea
minheap and whenever the minheap size reaches k, pop from minheap (so it will pop the smallest element each time)

at the end in the first position we will have the smallest element of k-sized array, that is the k largest element

the same applies to maxheap

if you want it to be optimized more than the heap approach, a *quickselect* is needed

**min heap:**
~~~py
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = []

        for n in nums:
            heapq.heappush(heap, n)

            if len(heap) > k:
                heapq.heappop(heap)

        return heap[0]
~~~

**max heap:**
~~~py
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = [-num for num in nums]

        heapq.heapify(heap)

        for _ in range(k - 1):
            heapq.heappop(heap)

        return -heap[0]
~~~

**complexity**
~~~
time = O(N + K logn) 
~~~
n = to turn the array in a heap and

k + logn = logn for each pop operation (to heapify) and k pop operations

~~~
space = O(N) 
~~~
for the heap size

### optimization

quickselect (TO DO)

## [227. Basic Calculator II](https://leetcode.com/problems/basic-calculator-ii/description)

### key idea
not using a stack due to extra space complexity

doing the operation iteratively, from left to right by keeping track of the previous and current number

as soon as we encounter * or /, we need to undo the previous + or - operation since the order of operations is / -> * -> +/-

remember to parse the numbers > 10 by doing the usual while loop

~~~py
class Solution:
    def calculate(self, s: str) -> int:
        cur = prev = res = 0
        op = "+"

        i = 0
        while i < len(s):
            char = s[i]
            if char.isdigit():
                # parsing numbers
                while i < len(s) and s[i].isdigit(): 
                    cur = cur * 10 + int(s[i])
                    i += 1
                i -= 1

                if op == "+":
                    res += cur
                    prev = cur
                elif op == "-":
                    res -= cur
                    prev = -cur
                elif op == "*":
                    res -= prev

                    res += prev * cur
                    prev = cur * prev
                elif op == "/":
                    res -= prev

                    # python has problems with negative numbers
                    #   so // will not work correctly
                    res += int(prev/cur)
                    prev = int(prev/cur)

                cur = 0
            elif char != " ":
                op = char

            i += 1
            
        return res
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through the whole string

~~~
space = O(1) 
~~~
no extra space, we happy

## [236. Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/description)

### key idea
if the left OR right subtree returns null, then it means that both p and q are in the subtree that didn't return null, and the node returned is the LCA

otherwise if both subtree return null, it means that the LCA is the parent of the left and right subtree

~~~py
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return None
        
        if root == p or root == q:
            return root
        
        l = self.lowestCommonAncestor(root.left, p, q) 
        r = self.lowestCommonAncestor(root.right, p, q) 

        if l and r:
            return root
        else:
            return l or r
~~~

**complexity**
~~~
time = O(N) 
~~~
in the worst case it checks the whole tree till the leaf nodes

~~~
space = O(N) 
~~~
n elements in the tree, so the size of the recursive stack is n, otherwise O(1)

### optimization

if you want to optimize it further watch this [video by errichto](https://www.youtube.com/watch?v=dOAxrhAUIhA&list=PLl0KD3g-oDOEbtmoKT5UWZ-0_JbyLnHPZ&index=17)

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
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        res = [1] * len(nums)

        prefix = 1
        for i in range(len(nums)):
            res[i] = prefix
            prefix *= nums[i]
        
        suffix = 1
        for i in range(len(nums) - 1, -1, -1):
            res[i] *= suffix
            suffix *= nums[i]
        
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

## 314. Binary Tree Vertical Order Traversal [(premium)](https://leetcode.com/problems/binary-tree-vertical-order-traversal/description) | [('premium')](https://www.lintcode.com/problem/651/)

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

        columns = defaultdict(list)
        queue = deque([(0, root)])

        while queue:
            col, node = queue.popleft()

            columns[col].append(node.val)

            if node.left:
                queue.append((col - 1, node.left))

            if node.right:
                queue.append((col + 1, node.right))

        res = dict(sorted(columns.items()))
        return list(res.values())
~~~

### optimization

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

## 339. Nested List Weight Sum [(premium)](https://leetcode.com/problems/nested-list-weight-sum/description)

### key idea
the difficulty of this problem is due to the interface implementation, because the algorithm is really straightforward

here it's the simple dfs solution

~~~py
class Solution:
    def depthSum(self, nestedList: List[NestedInteger]) -> int:
        def dfs(nested_list, depth) -> int:
            total = 0

            for nested in nested_list:
                if nested.isInteger():
                    total += nested.getInteger() * depth
                else:
                    total += dfs(nested.getList(), depth + 1)
            return total
            
        return dfs(nestedList, 1)
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through the whole list

~~~
space = O(N) 
~~~
recursive call stack worst case '[[[[[[]]]]]]'

### alternative (iterative) RECOMMENDED

if the interviewer alter the question by changing the constraint of the depth, a stack overflow might occur with the recursive approach (dfs), so its good to have in mind the iterative solution (bfs) as well

the key idea here is to spread out the list at the end of the queue each time one is encountered, so after every iteration we can go one level deeper

~~~py
class Solution:
    def depthSum(self, nestedList: List[NestedInteger]) -> int:
        depth = 1
        res = 0
        # spread out the whole list in the queue (the first level) for the bfs
        queue = deque(nestedList)

        while queue:
            for _ in range(len(queue)):
                item = queue.popleft()
                if item.isInteger():
                    res += item.getInteger() * depth
                else:
                    queue.extend(item.getList())
            
            depth += 1
        
        return res
~~~

**complexity**
~~~
time = O(N) 
~~~
same as dfs

~~~
space = O(N) 
~~~
same as dfs

## 346. Moving Average from Data Stream [(premium)](https://leetcode.com/problems/moving-average-from-data-stream) | [('premium')](https://www.lintcode.com/problem/642/)

### key idea

the naive solution is with a queue, but using a circular array is slightly better because:

- moving pointers, automatically discard the 'out of bound' elements, while with the queue we should do a deque operation each time
- we only need to store the pointer that points to the head and not on both side like the queue (head for popleft and tail for append)


so that's it basically, just remember to keep the size bounded to the requested size of the data stream

~~~py
class MovingAverage:

    def __init__(self, size: int):
        self.size = size
        self.circular_array = [0] * self.size
        self.head = 0
        self.count = 0
        self.res = 0

    def next(self, val: int) -> float:
        self.count += 1

        tail = (self.head + 1) % self.size
        self.res = self.res - self.circular_array[tail] + val

        self.head = (self.head + 1) % self.size
        self.circular_array[self.head] = val

        return self.res / min(self.size, self.count)
~~~

**complexity**
~~~
time = O(1) 
~~~
every call of next does only constant time operations

~~~
space = O(N) 
~~~
size of circular array

## [347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements)

### key idea
frequency count with hash map

since heap would lead us to O(k*logn) complexity, we use a slight variation of a technique called *bucket sort*

we construct an array based on the hash map that has the frequency count as index and a list of variables (with the same frequency) as a value

then we iterate the frequency array from right to left, to find the k top elements

~~~py
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = defaultdict(int) 
        freq = [[] for i in range(len(nums) + 1)]
        res = []

        # count numbers frequency
        for n in nums:
            count[n] += 1
        
        # construct array with count as indices and a list of number as value
        for num, cnt in count.items():
            freq[cnt].append(num)

        # iterate from left to right for k elements
        for i in range(len(freq) - 1, 0, -1):
            for n in freq[i]:
                res.append(n)
                if len(res) == k:
                    return res
~~~

**complexity**
~~~
time = O(N) 
~~~
we iterate through the whole array nums + construct the hash map (both O(n) operations)

~~~
space = O(N) 
~~~
for the hash map/freq array

## [398. Random Pick Index](https://leetcode.com/problems/random-pick-index/description)

### key idea

reservoir sampling

this technique let us choose a random index without using extra space complexity

~~~py
class Solution:

    def __init__(self, nums: List[int]):
        self.nums = nums

    def pick(self, target: int) -> int:
        count = pick_index = 0

        for i, n in enumerate(self.nums):
            if n == target:
                count += 1

                if random.randint(1, count) == count:
                    pick_index = i
        return pick_index
~~~

**complexity**
~~~
time = O(N) 
~~~
we iterate through all nums

~~~
space = O(1) 
~~~
no extra memory used

### resources

- [reservoir sampling article](https://florian.github.io/reservoir-sampling/)

## [415. Add Strings](https://leetcode.com/problems/add-strings/description)

### key idea

nothing to say other than i'm so bad

~~~py
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        res = deque()
        i = len(num1) - 1
        j = len(num2) - 1

        carry = 0
        while i >= 0 or j >= 0:
            cur_i = int(num1[i]) if i >= 0 else 0
            cur_j = int(num2[j]) if j >= 0 else 0

            cur_sum = carry + cur_i + cur_j

            res.appendleft(str(cur_sum % 10))

            carry = cur_sum // 10

            i -= 1
            j -= 1
        
        if carry:
            res.appendleft(str(carry))
        
        return "".join(res)
~~~

**complexity**
~~~
time = O(N) 
~~~
in the worst case len(num1) == len(num2) -> N + M -> N

~~~
space = O(1) or O(N) if we count result array 
~~~

## 426. Convert Binary Search Tree to Sorted Doubly Linked List [(premium)](https://leetcode.com/problems/convert-binary-search-tree-to-sorted-doubly-linked-list/description/) | [('premium')](https://www.lintcode.com/problem/1534/)

### key idea

in-order traversal

keep track of the first (so the most left value) and last (every return we modify this last, always in-order)

while doing the recursive calls, as operations, we need to modify the left (predecessor) and right (successor) pointers

at the end remember to link first and last

~~~py
class Solution:
    def treeToDoublyList(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if not root:
            return None

        first = None
        last = None

        def dfs(node):
            if not node:
                return 
            
            dfs(node.left)

            nonlocal first
            nonlocal last

            if not last:
                first = node
            else:
                last.right = node
                node.left = last

            last = node

            dfs(node.right)

        dfs(root)

        first.left = last
        last.right = first

        return first
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through the whole tree where n is the tree size

~~~
space = O(N) 
~~~
O(logN) if the tree is balanced, but in the worst case the tree is not balanced, so N is the size of recursive call stack

## [498. Diagonal Traverse](https://leetcode.com/problems/diagonal-traverse/description)

### key idea

here we need to follow the flow that the description suggested us

be careful with out of bounds

keep track if you are going up or down

~~~py
class Solution:
    def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
        row_length = len(mat)
        col_length = len(mat[0])
        res = []

        is_going_up = True
        cur_i = cur_j = 0
 
        while len(res) != row_length * col_length:
            if is_going_up:
                while cur_i >= 0 and cur_j < col_length:
                    res.append(mat[cur_i][cur_j])
                
                    cur_i -= 1
                    cur_j += 1
                
                if cur_j == col_length:
                    cur_i += 2
                    cur_j -= 1
                else:
                    cur_i += 1
                
                is_going_up = False
            else:
                while cur_i < row_length and cur_j >= 0:
                    res.append(mat[cur_i][cur_j])
                
                    cur_i += 1
                    cur_j -= 1

                if cur_i == row_length:
                    cur_i -= 1
                    cur_j += 2
                else:
                    cur_j += 1

                is_going_up = True
                
        return res
~~~

**complexity**
~~~
time = O(N * M) 
~~~
we process each element of the matrix

~~~
space = O(1) or O(N) if res is counted
~~~

## [523. Continuous Subarray Sum](https://leetcode.com/problems/continuous-subarray-sum/)

### key idea

hasmap and prefix sum

~~~
prefix[j] % k = prefix[i] % k  (with i < j)
~~~

store in the hashmap the reminder (key) and index (value), as soon a reminder is found, it means that two prefix sums have the same reminder and the last thing to check is if the length is greater than 1

example: 

[23, 2, 4, 3, 3] with k = 6

- 23 % 6 = 5
- 29 % 6 = 5
- 35 % 6 = 5

it means that all three make up a subarray of length 3 where the sum of all elements is a multiple of k, because we'll have:
- 23 - 29 = 6 (6 is a multiple of k)
- 23 - 35 = 12 (12 is a multiple of k)
- and obviously 29 - 35 = 6 (6 is a multiple of k)

~~~py
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        hmap = defaultdict(int)
        prefix_sum = 0
        hmap[0] = -1
        
        for i in range(len(nums)):
            prefix_sum += nums[i]

            if prefix_sum % k in hmap:
                if i - hmap[prefix_sum % k] > 1:
                    return True
            else:
                hmap[prefix_sum % k] = i
        
        return False
~~~

**complexity**
~~~
time = O(N) 
~~~
we iterate only once

~~~
space = O(N) 
~~~
hash map size

## [525. Contiguous Array](https://leetcode.com/problems/contiguous-array)

### key idea

count increment/decrement if we see a 1/0

store the count in a dictionary count:index

compute the difference between the current index and the dictionary count index

~~~py
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        count_dict = defaultdict(int)
        count_dict[0] = -1
        count = 0
        res = 0

        for i, n in enumerate(nums):
            if n == 0:
                count -= 1
            else:
                count += 1

            if count in count_dict:
                res = max(res, i - count_dict[count])
            else:
                count_dict[count] = i
        
        return res
~~~

**complexity**
~~~
time = O(N) 
~~~
in the worst case all the array is the maximum contiguous array

~~~
space = O(N) 
~~~
dictionary size

## [528. Random Pick with Weight]()

### key idea

this problem has a tricky intuition and the problem statement is written by monkeys

given an array of weights corresponding to their indexes, as the weight increases, the probabilty of randomly picking that specific weight's index also increases

basically, the problem is saying: "randomly pick an index but take its weight into account"

to do this we utilize the concept of a cumulative distribution function (prefix sum)

it allows us to get intervals, and these intervals simulate the weight for their specific index by considering their length.

'target = random.uniform(0, self.maxValue)' is used to get a random variable that is uniformly distributed, and thanks to this uniform distribution, we can say that the probability of this target being chosen is based on the length of the intervals created by the cdf

the solution is actually simple to remember

~~~py
class Solution:
    def __init__(self, w: List[int]):
        self.prefix_sum = []

        total = 0

        for weight in w:
            total += weight

            self.prefix_sum.append(total)

        self.maxValue = 0

    def pickIndex(self) -> int:
        target = random.uniform(0, self.maxValue)

        l = 0
        r = len(self.prefix_sum)

        while l < r:
            mid = (l + r) // 2

            if self.prefix_sum[mid] < target:
                l = mid + 1
            else:
                r = mid
        
        return l
~~~

**complexity**
~~~
time = init: O(N) - pickIndex: O(logN)
~~~

~~~
space = init: O(N) - pickIndex: O(1)
~~~

### resources

- [video explanation](https://www.youtube.com/watch?v=7x7Ydq2Wfvw)

## [543. Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree)

### key idea
the thing to remember for this problem that i always forget is that you need to add 1 at the end (to count edges) and not for every dfs call (to count nodes)

plus the diameter could be:

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

## [560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k)

### key idea

we use prefix sum because the difference between two prefix sums gives us the sum of the subarray between those two indices:

~~~
prefix_sum_i - prefix_sum_j = subarray_sum_ij = k
~~~

let's use this concept to find efficiently how many subarray will give k as our result:

~~~
prefixsum_i - k = prefixsum_j where i > j
~~~

we can calculate prefix sums and count those using an hashmap

then whenever 'prefixsum - k' is in the hashmap, it means that we have found a new subarray

easy

~~~py
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        prefix_dict = defaultdict(int)
        prefix_sum = res = 0

        # this is important for some edge cases to count the very first subarray
        prefix_dict[0] = 1

        for num in nums:
            prefix_sum += num

            if prefix_sum - k in prefix_dict:
                res += prefix_dict[prefix_sum - k]
            prefix_dict[prefix_sum] += 1
        return res
~~~

**complexity**
~~~
time = O(N) 
~~~
we iterate the whole array nums

~~~
space = O(N) 
~~~
for the hashmap

## [636. Exclusive Time of Functions](https://leetcode.com/problems/exclusive-time-of-functions)

### key idea

as the problem suggest, use a stack

keep track of the execution times using a simple array and indexing by id

compute the delta between the previous time and the current time

remember that the end time is inclusive so '+ 1' is needed

~~~py
class Solution:
    def exclusiveTime(self, n: int, logs: List[str]) -> List[int]:
        stk = []
        res = [0] * n
        prev_time = 0

        for l in logs:
            log_id, log_startend, log_time = l.split(":")
            log_id = int(log_id)
            log_time = int(log_time)

            if log_startend == "start":
                if stk:
                    res[stk[-1]] += log_time - prev_time
                
                stk.append(log_id)
                prev_time = log_time
            else:
                res[stk.pop()] += (log_time - prev_time) + 1
                prev_time = log_time + 1
        
        return res
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through the whole logs

~~~
space = O(N) 
~~~
stack/array space

## 637. Valid Word Abbreviation [(premium)](https://leetcode.com/problems/valid-word-abbreviation/description) | [('premium')](https://www.lintcode.com/problem/637/)

### key idea
the idea is simple, two pointers, *word_ptr* and *abbr_ptr*

whenever a number is encountered in the abbr, process the steps based on the number and increment the word_ptr by these steps

but the execution is BULLSHIT, too many edge cases and not an easy question

~~~py
class Solution:
    def validWordAbbreviation(self, word: str, abbr: str) -> bool:
        word_ptr, abbr_ptr = 0, 0

        while word_ptr < len(word) and abbr_ptr < len(abbr):
            if abbr[abbr_ptr].isdigit():
                if abbr[abbr_ptr] == "0":
                    return False
                
                step = 0
                while abbr_ptr < len(abbr) and abbr[abbr_ptr].isdigit():
                    step = step * 10 + int(abbr[abbr_ptr])
                    abbr_ptr += 1
                
                word_ptr += step
            else:
                if word[word_ptr] != abbr[abbr_ptr]:
                    return False
                
                word_ptr += 1
                abbr_ptr += 1
            
        return word_ptr == len(word) and abbr_ptr == len(abbr)
~~~

**complexity**
~~~
time = O(N) 
~~~
both word and abbr are of size n

~~~
space = O(1) 
~~~
we only use pointers

## [647. Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings)

### key idea

same as [5. Longest Palindromic Substring](#5-longest-palindromic-substring) but instead of calculating the result, just increase the count

~~~py
class Solution:
    def countSubstrings(self, s: str) -> int:
        count = 0
        
        for i in range(len(s)):
            count += self.count_palindromes(s, i, i)
            count += self.count_palindromes(s, i, i+1)

        return count
    
    def count_palindromes(self, s, l, r):
            count = 0
            while l >= 0 and r < len(s) and s[l] == s[r]:
                l -= 1
                r += 1
                count += 1

            return count
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through the whole string

~~~
space = O(1) 
~~~
no extra space used

## [670. Maximum Swap](https://leetcode.com/problems/maximum-swap)

### key idea

so basically the idea is to search the rightmost gratest value (so from right to left) and swap this value with the first leftmost smallest value (from left to right)

the code is a little tricky but it's fun because we are parsing back and forth the number

~~~py
class Solution:
    def maximumSwap(self, num: int) -> int:
        if num <= 11:
            return num
        
        # convert the integer number in an array
        num_as_arr = deque()
        while num:
            num_as_arr.appendleft(num % 10)
            num //= 10
        
        # get the rightmost largest value for each number
        max_seen, max_seen_at = -1, len(num_as_arr)
        for i in range(len(num_as_arr) - 1, -1, -1):
            cur_num = num_as_arr[i]
            num_as_arr[i] = (cur_num, max_seen, max_seen_at)

            if cur_num > max_seen:
                max_seen = cur_num
                max_seen_at = i
        
        # find the first smallest value and swap it with the rightmost largest value
        for i in range(len(num_as_arr)):
            cur_num, max_seen, max_seen_at = num_as_arr[i]

            if cur_num < max_seen:
                num_as_arr[i], num_as_arr[max_seen_at] = num_as_arr[max_seen_at], num_as_arr[i]
                break
        
        # convert the array back into an integer
        num = 0
        for n, _, _ in num_as_arr:
            num = num * 10 + n
        
        return num
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through the number, value by value

~~~
space = O(N) 
~~~
number array size


## [680. Valid Palindrome II](https://leetcode.com/problems/valid-palindrome-ii)

### key idea
*two pointers*

whenever one of both sides is not equal, check if by removing one of the two sides it will become palindrome, by comparing the string without the left character and the string without the right character using its reverse (the code is self-explanatory with some python knowledge)

**remember** to check both possibilites, so skip left, check, skip right, check

~~~py
class Solution:
    def validPalindrome(self, s: str) -> bool:
        l, r = 0, len(s) - 1

        while l < r:
            if s[l] != s[r]:
                skipLeft, skipRight = s[l + 1:r + 1], s[l:r]

                # palindrome check with reverse technique
                # if both are false it means that even with deleting the character
                #   the string is still not palindrome
                return (skipLeft == skipLeft[::-1]) or (skipRight == skipRight[::-1])
            
            l += 1
            r -= 1

        return True
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through all the string

~~~
space = O(N) 
~~~
for skipLeft/skipRight

## optimization

we can optimize this problem by eliminating skipLeft/skipRight and reversing

by using only pointers

## [708. Insert into a Sorted Circular Linked List](https://leetcode.com/problems/insert-into-a-sorted-circular-linked-list)

### key idea

we must take in account 3 cases:

1) head is null *(create the new node and link it to himself)*
2) the node needs to be inserted between prev and next *(as soon as the node value is in between prev and next)*
3) the node needs to be inserted after the tail *(whenever cur.val > cur.next.val we got to the tail)*

~~~py
class Solution:
    def insert(self, head: 'Optional[Node]', insertVal: int) -> 'Node':
        if not head:
            new_node = Node(insertVal, None)
            new_node.next = new_node
            return new_node

        cur = head

        while cur.next != head:
            if cur.val <= insertVal <= cur.next.val:
                new_node = Node(insertVal, cur.next)
                cur.next = new_node
                return head
            elif cur.val > cur.next.val: # tail
                if insertVal >= cur.val or insertVal <= cur.next.val:
                    new_node = Node(insertVal, cur.next)
                    cur.next = new_node
                    return head
            
            cur = cur.next
        
        new_node = Node(insertVal, cur.next)
        cur.next = new_node

        return head
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through the whole linked list

~~~
space = O(1) 
~~~
no extra memory used

## [721. Accounts Merge](https://leetcode.com/problems/accounts-merge/description)

### key idea

graph problem

make connections between emails of the same account (for efficiency connect everything to the first email)

then to merge the accounts we just traverse the graph

remember to:

- keep a mapping email -> name for the result
- sort the emails from the traversal

~~~py
class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        graph = defaultdict(set)
        email_to_name = {}

        for acc in accounts:
            name = acc[0]

            for email in acc[1:]:
                graph[email].add(acc[1])
                graph[acc[1]].add(email)

                email_to_name[email] = name
        
        visited = set()
        res = []

        def dfs(node, local_res):
            visited.add(node)

            local_res.append(node)

            for adj in graph[node]:
                if adj not in visited:
                    dfs(adj, local_res)
        
        for src in list(graph):
            if src not in visited:
                local_res = []
                dfs(src, local_res)
                res.append([email_to_name[src]] + sorted(local_res))
        
        return res
~~~

**complexity**
~~~
time = O(NKlogNK) 
~~~
n number of accounts and k maximum length of an account

in the worst case all email belongs to a single person + there is sorting

~~~
space = O(NK) 
~~~
graph size, visited size, recursive call stack size, map size

## [767. Reorganize String](https://leetcode.com/problems/reorganize-string)

### key idea

use an hashmap to count letters

for each iteration, at the start always pick the letter with the greatest count then pick the other letters accordingly

to pick the greatest we gonna use a max heap instead of iterating the hashmap

~~~py
class Solution:
    def reorganizeString(self, s: str) -> str:
        count = Counter(s)
        max_heap = [[-cnt, char] for char, cnt in count.items()]

        prev = None
        res = ""

        while max_heap or prev:
            if not max_heap and prev:
                return ""

            cnt, char = heappop(max_heap)
            res += char
            cnt += 1

            if prev:
                heappush(max_heap, prev)
                prev = None

            if cnt != 0:
                prev = [cnt, char]
        
        return res
~~~

**complexity**
~~~
time = O(NlogN) 
~~~
heap operations

~~~
space = O(N) 
~~~
we store the whole string in the heap


## [791. Custom Sort String](https://leetcode.com/problems/custom-sort-string)

### key idea

hashmap that counts the chars in s

for each char in order, if it's also present in the hashmap build the string with n times that char

at the end the string s could be left with some extra characters not present in order, so iterate through the remaining char in the hashmap and construct the final result

the code is pretty self explanatory

~~~py
class Solution:
    def customSortString(self, order: str, s: str) -> str:
        map_char_count = Counter(s)
        res = ""

        for o in order:
            res += (o * map_char_count[o])
            del map_char_count[o]
        
        for char, count in map_char_count.items():
            res += (char * count)
        
        return res
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through all the string + counter operation

~~~
space = O(N) 
~~~
counter size

## [827. Making A Large Island](https://leetcode.com/problems/making-a-large-island)

### key idea

first calculate the area of connected 1 islands and mark each island with a unique island_id

these islands will be store in an hashmap

after this, we go through all the 0s and check for surrounding islands, if there is a surrounding island then add its area (using the hashmap and island_id)

after all 0s are computed, we've found the largest island with at most one 0

~~~py
class Solution:
    def largestIsland(self, grid: List[List[int]]) -> int:
        island_id = -1
        island_areas = {}
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        def compute_area(i, j):
            if (0 <= i < len(grid)) and (0 <= j < len(grid[0])) and grid[i][j] == 1:
                grid[i][j] = island_id

                area = 1
                for r_inc, c_inc in directions:
                    new_i = i + i_inc
                    new_j = j + j_inc
                    area += compute_area(new_i, new_j)
                
                return area
            else:
                return 0

        # compute the areas of all islands and assign them an id
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    area = compute_area(i, j)

                    island_areas[island_id] = area
                    island_id -= 1
        
        max_area = 0

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 0:
                    area = 1

                    # build the surrounding
                    surrounding = set()
                    for i_inc, j_inc in directions:
                        new_i = i + i_inc
                        new_j = j + j_inc

                        if (0 <= new_i < len(grid)) and (0 <= new_j < len(grid[0]) and grid[new_i][new_j] != 0):
                            surrounding.add(grid[new_i][new_j])
                     
                    # use the surrounding of the 0 to compute the whole area
                    for island_id in surrounding:
                        area += island_areas[island_id]
                    
                    max_area = max(max_area, area)
        
        # there could happen a case where the whole grid is 1
        return max_area if max_area else len(grid) ** 2
~~~

**complexity**
~~~
time = O(N^2) 
~~~
N * N size of the matrix, so it's the iterations through all the matrix

~~~
space = O(N^2 / 2) 
~~~
this is the worst case if the matrix is half 0 and half 1

## [875. Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas)

### key idea

create a range and search through it

the minimum banana that koko can eat is 1, and the maximum is the maximum value of the piles

now we need something to search between this range, and check whetever we find a valid value that will make koko eat n bananas in exactly h hours

usually when we talk of searching in a sorted array for a **specific condition** we do a binary search with the caveaut that it will run **while l < r** and we'll **return the left pointer** as our result

~~~py
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        l = 1
        r = max(piles)

        def can_eat(speed):
            hours = 0

            for p in piles:
                if speed > p:
                    hours += 1
                else:
                    hours += math.ceil(p / speed)
            
            return hours <= h

        while l < r:
            mid = (l + r) // 2

            if not can_eat(mid):
                l = mid + 1
            else:
                r = mid

        return l
~~~

**complexity**
~~~
time = O(NlogN) 
~~~
for each binary search we execute can_eat that takes n time

~~~
space = O(1) 
~~~
no extra space used

## [921. Minimum Add to Make Parentheses Valid](https://leetcode.com/problems/minimum-add-to-make-parentheses-valid/description)

### key idea

check for extra right parentheses and whenever one is found increment a counter that counts how many open parentheses do we need to add

~~~py
class Solution:
    def minAddToMakeValid(self, s: str) -> int:
        left_count = right_count = added = 0

        for c in s:
            if c == "(":
                left_count += 1
            else:
                if right_count < left_count:
                    right_count += 1
                else:
                    added += 1
        
        added += left_count - right_count
        
        return added
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through all the string

~~~
space = O(1) 
~~~
only counters, no extra space used

## [938. Range Sum of BST](https://leetcode.com/problems/range-sum-of-bst)

### key idea
whenever you are out of range, return accordingly

the return is used to go right or left without computing the sum at the end, so it's a sort of roadblock

we compute the sum only if we are inside the range

~~~py
class Solution:
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        def dfs(node) -> int:
            if not node:
                return 0
            
            if node.val < low:
                return dfs(node.right)
                
            if node.val > high:
                return dfs(node.left)

            return node.val + dfs(node.left) + dfs(node.right)
        
        return dfs(root)
~~~

**complexity**
~~~
time = O(N) 
~~~
in the worst case we'll visit all the tree (if low and high are at the edge of the tree)

~~~
space = O(N) 
~~~
recursive call stack size since in the worst case we recurse n times (how many nodes there are)

## [953. Verifying an Alien Dictionary](https://leetcode.com/problems/verifying-an-alien-dictionary)

### key idea

create an order dictionary that stores order_char : order_index

that way we can check if for each word, some word is in the wrong order

compare the words pairwise and for each pair, check if they are in the correct order

~~~py
class Solution:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        order_dict = {char : i for i, char in enumerate(order)}

        def are_words_ordered(word1, word2):
            for i in range(min(len(word1), len(word2))):
                if word1[i] != word2[i]:
                    if order_dict[word1[i]] > order_dict[word2[i]]:
                        return False
                    else:
                        return True
            
            return len(word1) <= len(word2)

        for word1, word2 in zip(words, words[1:]):
            if not are_words_ordered(word1, word2):
                return False

            return True
~~~

**complexity**
~~~
time = O(26 + 100) -> O(1)
~~~
to construct the dictionary it take O(26) cause there are 26 letters in order

words.length <= 100

~~~
space = O(1) 
~~~
same as time complexity

## [973. K Closest Points to Origin]()

### key idea
use a min heap, the closest points to the origin are the one at the start of the min heap

~~~py
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        minHeap = []

        for point in points:
            dist = point[0] ** 2 + point[1] ** 2
            heapq.heappush(minHeap, (dist, point))
        
        res = []
        for _ in range(k):
            res.append(heapq.heappop(minHeap)[1])
        
        return res
~~~

**complexity**
~~~
time = O(N log N + K log N) 
~~~
build the heap and process every point

~~~
space = O(N) 
~~~
heap takes n points

## [986. Interval List Intersections](https://leetcode.com/problems/interval-list-intersections/description)

### key idea

two pointers

look at example diagram / draw it yourself, then try to come up with all the possibles outcomes and code them out

~~~py
class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        if not firstList or not secondList:
            return []
            
        p1, p2 = 0, 0
        res = []

        while p1 < len(firstList) and p2 < len(secondList):
            start1, start2 = firstList[p1][0], secondList[p2][0]
            end1, end2 = firstList[p1][1], secondList[p2][1]

            if start1 > end2: # check if they are disjoint
                p2 += 1
            elif start2 > end1:
                p1 += 1
            else: # if they are overlapping
                res.append([max(start1, start2), min(end1, end2)])

                if end1 > end2: # check if there is some processing left
                    p2 += 1
                else:
                    p1 += 1
        return res
~~~

**complexity**
~~~
time = O(N + M)
~~~
if both l1.length = N and l2.length = M are of the same size

~~~
space = O(1) or O(N) if considering the res array
~~~

## [994. Rotting Oranges](https://leetcode.com/problems/rotting-oranges)

### key idea

bfs, keep count of the fresh oranges, push in the queue rotten oranges

that's it

~~~py
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        q = deque()
        directions = [(1,0), (0,1), (-1,0), (0,-1)]
        time = fresh = 0

        row_length = len(grid)
        col_length = len(grid[0])
        for i in range(row_length):
            for j in range(col_length):
                if grid[i][j] == 1:
                    fresh += 1
                if grid[i][j] == 2:
                    q.append([i, j])
        
        while q and fresh > 0:
            cur_q_len = len(q)
            for _ in range(cur_q_len):
                cur_i, cur_j = q.popleft()

                for i_inc, j_inc in directions:
                    new_i = cur_i + i_inc
                    new_j = cur_j + j_inc

                    if 0 <= new_i < row_length and 0 <= new_j < col_length and grid[new_i][new_j] == 1:
                        grid[new_i][new_j] = 2
                        q.append([new_i, new_j])
                        fresh -= 1
            time += 1
        
        return time if fresh == 0 else -1
~~~

**complexity**
~~~
time = O(M * N)
~~~
iteration throught the whole matrix

~~~
space = O(M * N)
~~~
the queue in the worst case will be the whole matrix

## [1004. Max Consecutive Ones III](https://leetcode.com/problems/max-consecutive-ones-iii)

### key idea

greedy sliding window approach

if we encounter a 0, we decrement k

if k goes negative, that means that we took too many 0s, so we must move the left pointer up and we need to retake one 0 each time

check the maximum of all the results

~~~py
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        l = res = 0

        for r, num in enumerate(nums):
            # if num == 0:
            #     k -= 1
            k -= 1 - num
            
            if k < 0:
                # if nums[l] == 0:
                #   k += 1
                k += 1 - nums[l]
                
                l += 1
            else:
                res = max(res, (r - l) + 1)

        return res
~~~

**complexity**
~~~
time = O(N) 
~~~
only one iteration

~~~
space = O(1) 
~~~
only pointers, no extra space used

## [1091. Shortest Path in Binary Matrix](https://leetcode.com/problems/shortest-path-in-binary-matrix)

### key idea

bfs to find the shortest path

remember that you can go diagonally so there are 8 possible directions

the bfs queue stores (row, column, length) and as soon row and column reach N - 1, return the length

~~~py
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        queue = deque([(0,0,1)])
        directions = [[1,0], [0,1], [-1,0], [0,-1], [1,1], [-1,1], [1,-1], [-1,-1]]
        visited = set((0,0))

        while queue:
            r, c, length = queue.popleft()

            if (min(r, c) < 0 or max(r, c) == len(grid) or grid[r][c] == 1):
                continue
            
            if r == len(grid) - 1 and c == len(grid) - 1:
                return length
            
            for row_dir, col_dir in directions:
                new_row = r + row_dir
                new_col = c + col_dir
                if (new_row, new_col) not in visited:
                    queue.append((new_row, new_col, length + 1))
                    visited.add((new_row, new_col))
        
        return -1
~~~

**complexity**
~~~
time = O(N * M) 
~~~
we visit all the 2d matrix

~~~
space = O(N * M) 
~~~
we might have the entire matrix in the queue in the worst case

## [1249. Minimum Remove to Make Valid Parentheses](https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses)

### key idea
keep a count of open parentheses and whenever we encounter a closing parentheses we decrement the count

there is a little evil edgecase where there could be no closing parentheses but only open parentheses and that's why we have two iterations:

1) iteration to remove extra closing parentheses
2) iteration to remove the most right extra open parentheses (that's why we iterate the reversed array)

~~~py
class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        res = []
        openCount = 0

        # skip extra closing parentheses
        for c in s:
            if c == "(":
                res.append(c)
                openCount += 1
            elif c == ")" and openCount > 0:
                res.append(c)
                openCount -= 1
            elif c != ")":
                res.append(c)
        
        filtered = []

        # skip extra most right open parentheses
        for c in res[::-1]:
            if c == "(" and openCount > 0:
                openCount -= 1
            else:
                filtered.append(c)
        
        return "".join(filtered[::-1])
~~~

**complexity**
~~~
time = O(N) 
~~~
two indipendent iterations that goes only through all the elements of the string

~~~
space = O(N) 
~~~
res/filtered size is at most n

## [1539. Kth Missing Positive Number](https://leetcode.com/problems/kth-missing-positive-number/description)

### key idea

we will use a binary search on indexes to find the missing number in an efficient way

by doing arr[i] - i we can get the numbers of missing numbers up till that point

so with this information we can find where the missing number should be in O(logN) time thanks to binary search

to clarify better this concept just watch the video in the resources

~~~py
class Solution:
    def findKthPositive(self, arr: List[int], k: int) -> int:
        l, r = 0, len(arr) - 1

        while l <= r:
            mid = (l + r) // 2

            if arr[mid] - mid - 1 < k:
                l = mid + 1
            else:
                r = mid - 1
        
        return l + k
~~~

**complexity**
~~~
time = O(logN) 
~~~
binary searchhhh

~~~
space = O(1) 
~~~
no extra memory

### resources

- [explanation video](https://www.youtube.com/watch?v=NObPmjZIh8Y)

## 1570. Dot Product of Two Sparse Vectors [(premium)](https://leetcode.com/problems/dot-product-of-two-sparse-vectors/description) | [('premium')](https://www.lintcode.com/problem/3691/)

### key idea
there are actually 3 solutions:

- naive: store the entire array and do the 1 by 1 product
- hashmap: don't store the whole array by storing only the non-zero values as index -> value
- tuple array and two pointers: array of tuples (idx, val) with two pointers, one for each vector

the tuple array is the most efficient one since there could be problems if the hash functions sucks basically

really easy to implement

~~~py
class SparseVector:
    def __init__(self, nums: List[int]):
        self.tuple_arr = []

        for i, n in enumerate(nums):
            if n != 0:
                self.tuple_arr.append((i, n))

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: 'SparseVector') -> int:
        res = 0
        p1 = p2 = 0       

        while p1 < len(self.tuple_arr) and p2 < len(vec.tuple_arr):
            p1_idx, p1_val = self.tuple_arr[p1]
            p2_idx, p2_val = vec.tuple_arr[p2]

            if p1_idx == p2_idx:
                res += p1_val * p2_val
                p1 += 1
                p2 += 1
            else:
                if p1_idx > p2_idx:
                    p2 += 1
                else:
                    p1 += 1
        return res
~~~

**complexity**
~~~
time = O(N + M) 
~~~
in the worst case the idx wont match and a full iteration for both has to be done

~~~
space = O(N + M) 
~~~
we store two tuple arrays

## 1650. Lowest Common Ancestor of a Binary Tree III [(premium)](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iii)

### key idea
not the traditional solution explained in the tutorials, but i like this one

basically is SFS (Swim First Search)

based on the distance from the top, if the two distances are different, then the one that is lower need to swim till the one higher, as soon as they are at the same height then they swim together until the LCA

~~~py
class Solution:
    def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node':
        def distFromTop(node) -> int:
            if not node:
                return -1
            return 1 + distFromTop(node.parent)
        
        pDistFromTop = distFromTop(p)
        qDistFromTop = distFromTop(q)

        while pDistFromTop > qDistFromTop:
            p = p.parent
            pDistFromTop -= 1
        while qDistFromTop > pDistFromTop:
            q = q.parent
            qDistFromTop -= 1
        
        while p != q:
            p = p.parent
            q = q.parent
        
        return p
~~~


**complexity**
~~~
time = O(N) 
~~~
we go through the whole tree in the worst case

~~~
space = O(1) 
~~~
no extra space used, only iterations (if we don't count the call stack to go to the top)

## 1762. Buildings With an Ocean View [(premium)](https://leetcode.com/problems/buildings-with-an-ocean-view/description)

### key idea
easy but marked medium, just watch for the tallest building with a ptr, skip every smaller building and just insert in the result array the indexes of the higher buildings. the code is self explanatory

we use a deque to save time by appending at the left instead of reversing the array at the end

~~~py
class Solution:
    def findBuildings(self, heights: List[int]) -> List[int]:
        if not heights:
            return []

        answer = deque([])
        max_height = -1

        for i in range(len(heights) - 1, -1, -1):
            cur = heights[i]

            if cur > max_height:
                answer.appendleft(i)

                max_height = cur

        return answer
~~~

**complexity**
~~~
time = O(N) 
~~~
we go through all the buildings in the worst case

~~~
space = O(N) 
~~~
we need to store n elements in res array in the worst case

### alternative (from left to right)

it might happen that the interviewer will change the constraint for example you are only allowed to go from left to right, then use this solution with a stack

~~~py
class Solution:
    def findBuildings(self, heights: List[int]) -> List[int]:
        n = len(heights)
        answer = []

        for cur_idx in range(n):
            while answer and heights[answer[-1]] <= heights[cur_idx]:
                answer.pop()
            answer.append(cur_idx)
        
        return answer
~~~

**complexity**
~~~
time = O(N) 
~~~
same as before

~~~
space = O(N) 
~~~
same as before

## 1868. Product of Two Run Length Encoded Arrays [(premium)](https://leetcode.com/problems/product-of-two-run-length-encoded-arrays) | [('premium')](https://www.lintcode.com/problem/3730/)

### key ide

the idea is to process 'in place' the product without creating extra arrays, using two pointers

if the last appended product in res is still the current product processed, then increment its relative frequency since we want the most optimized run-length algorithm

move the pointers accordingly based on the minimum frequency

~~~py
class Solution:
    def findRLEArray(self, encoded1: List[List[int]], encoded2: List[List[int]]) -> List[List[int]]:
        p1 = p2 = 0
        res = []

        while p1 < len(encoded1) and p2 < len(encoded2):
            num1, count1 = encoded1[p1]
            num2, count2 = encoded2[p2]

            prod = num1 * num2
            freq = min(count1, count2)

            if not res or prod != res[-1][0]:
                res.append([prod, freq])
            else:
                res[-1][1] += freq # add the frequency and optimize the run-length encoded array
            
            # consume the used counts
            encoded1[p1][1] -= freq
            encoded2[p2][1] -= freq

            # increment based on the smallest counter
            if count1 == freq:
                p1 += 1
            if count2 == freq:
                p2 += 1
                
        return res
~~~

**complexity**
~~~
time = O(N + M) 
~~~
N = encoded1 length

M =  encoded2 length

we iterate through encoded1 and encoded2

~~~
space = O(N + M) 
~~~
same as time complexity

## [2055. Plates Between Candles](https://leetcode.com/problems/plates-between-candles/)

### key idea

prefix sum to count the plates ("*")

left and right candles ("|") array to keep track of first candle position to the left and to the right

to calculate the result just use the property of the prefix sum to get the number of candles between the two candles by doing the difference

there is an edge case where start will be after the end position, in that case return 0 and it means that there are no candles

~~~py
class Solution:
    def platesBetweenCandles(self, s: str, queries: List[List[int]]) -> List[int]:
        prefix_sum = [0] * len(s)
        right_candles = [0] * len(s)
        left_candles = [0] * len(s)

        # calculate candles prefix_sum
        prefix_sum[0] = 1 if s[0] == "*" else 0
        for i in range(1, len(s)):
            prefix_sum[i] = prefix_sum[i - 1] + (1 if s[i] == "*" else 0)

        # calculate left candles positions
        left_candles[0] = 0 if s[0] == "|" else -1
        for i in range(1, len(s)):
            left_candles[i] = i if s[i] == "|" else left_candles[i - 1]

        # calculate right candles positions
        right_candles[len(s) - 1] = len(s) - 1 if s[len(s) - 1] == "|" else len(s)
        for i in range(len(s) - 2, -1, -1):
            right_candles[i] = i if s[i] == "|" else right_candles[i + 1]

        # calculate the result based on the queries
        res = [0] * len(queries)
        for i in range(len(queries)):
            start = right_candles[queries[i][0]]
            end = left_candles[queries[i][1]]

            res[i] = 0 if start >= end else prefix_sum[end] - prefix_sum[start]
        
        return res
~~~

**complexity**
~~~
time = O(N) 
~~~
we iterate through the whole string 

~~~
space = O(N) 
~~~
len(s) = n

prefix_sum and right/left candles are of size n

## 2340. Minimum Adjacent Swaps to Make a Valid Array [(premium)](https://leetcode.com/problems/minimum-adjacent-swaps-to-make-a-valid-array)

### key idea

find the rightmost largest element index and leftmost smallest element index

calculate the swaps by doing a math calculation

there is a small edge case: if the smallest element is at the right of the largest element we need to subtract 1 from the result since one swap is needed by both sides, a small advantage let's say

~~~py
class Solution:
    def minimumSwaps(self, nums: List[int]) -> int:
        max_value_pos = len(nums) - 1
        min_value_pos = 0

        for i, n in enumerate(nums):
            if nums[min_value_pos] > n:
                min_value_pos = i
            if nums[max_value_pos] <= n: # the = is needed to get the rightmost largest value index
                max_value_pos = i
            
        res = (len(nums) - 1 - max_value_pos) + min_value_pos
        
        return res - 1 if min_value_pos > max_value_pos else res
~~~

**complexity**
~~~
time = O(N) 
~~~
in the worst case we go through the whole array

~~~
space = O(1) 
~~~
no extra memory used
