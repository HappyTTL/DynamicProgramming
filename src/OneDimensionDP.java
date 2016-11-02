import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Created by Tingting on 10/3/16.
 */
public class OneDimensionDP {
    /**
     * Unique Binary Search Trees: Given n, how many structurally unique BST's (binary search trees) that store values
     * 1...n
     *    1        1        2       3       3
     *     \        \      / \     /       /
     *     3        2     1  3    2       1
     *    /          \           /        \
     *   2           3          1         2
     *   比如，以1为根的树有几个，完全取决于有二个元素的子树有几种。同理，2为根的子树取决于一个元素的子树有几个。以3为根的情况，则与1相同。

     定义Count[i] 为以[0,i]能产生的Unique Binary Tree的数目，

     如果数组为空，毫无疑问，只有一种BST，即空树，
     Count[0] =1

     如果数组仅有一个元素{1}，只有一种BST，单个节点
     Count[1] = 1
     Count[2] = Count[0] * Count[1]   (1为根的情况)
     + Count[1] * Count[0]  (2为根的情况。

     再看一遍三个元素的数组，可以发现BST的取值方式如下：
     Count[3] = Count[0]*Count[2]  (1为根的情况): 左子树0个node,右子树两个
     + Count[1]*Count[1]  (2为根的情况): 左子树1个,右子树1个
     + Count[2]*Count[0]  (3为根的情况): 左子树2个,右子树0个

     所以，由此观察，可以得出Count的递推公式为
     Count[i] = ∑ Count[0...k] * [ k+1....i]     0<=k<i-1
     问题至此划归为一维动态规划。
     */
    public int numTrees(int n) {
        int[] count = new int[n + 1];
        count[0] = 0;
        count[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                count[i] += count[j] * count[i - j - 1];
            }
        }
        return count[n];
    }

    /**
     * House Robber III: The thief has found himself a new place for his thievery again. There is only one entrance to
     * this area, called the "root." Besides the root, each house has one and only one parent house. After a tour, the
     * smart thief realized that "all houses in this place forms a binary tree". It will automatically contact the
     * police if two directly-linked houses were broken into on the same night.

     Determine the maximum amount of money the thief can rob tonight without alerting the police.

     初始化一个二维数组, 第一个元素表示该root 被rob, 第二个元素表示not rob
     f[0] = Math.max(fLeft[0], fLeft[1]) + Math.max(fRight[0], fRight[1])
     f[1] = fLeft[0] + fRight[0] + root.val;
     Initialization f[0] = 0, f[1] = 0;
     */
    public int rob(TreeNode root) {
        if (root == null) return 0;
        int[] result = robHelper(root);
        return Math.max(result[0], result[1]);
    }

    private int[] robHelper(TreeNode root) {
        int[] dp = {0,0};
        if (root != null) {
            int[] dpLeft = robHelper(root.left);
            int[] dpRight = robHelper(root.right);
            dp[0] = Math.max(dpLeft[0], dpLeft[1]) + Math.max(dpRight[0], dpRight[1]);
            dp[1] = dpLeft[0] + dpRight[0] + root.val;
        }
        return dp;
    }

    /**
     * Dungeon Game: The demons had captured the princess (P) and imprisoned her in the bottom-right corner of a
     * dungeon. The dungeon consists of M x N rooms laid out in a 2D grid. Our valiant knight (K) was initially
     * positioned in the top-left room and must fight his way through the dungeon to rescue the princess.

     The knight has an initial health point represented by a positive integer. If at any point his health point drops
     to 0 or below, he dies immediately.

     Some of the rooms are guarded by demons, so the knight loses health (negative integers) upon entering these rooms;
     other rooms are either empty (0's) or contain magic orbs that increase the knight's health (positive integers).

     In order to reach the princess as quickly as possible, the knight decides to move only rightward or downward in
     each step.


     Write a function to determine the knight's minimum initial health so that he is able to rescue the princess.

     For example, given the dungeon below, the initial health of the knight must be at least 7 if he follows the
     optimal path RIGHT-> RIGHT -> DOWN -> DOWN.

     -2(K)	-3	3
     -5	   -10	1
     10	   30	-5(P)
     dp[i][j]表示进入这个格子后保证knight不会死所需要的最小HP。如果一个格子的值为负，那么进入这个格子之前knight需要有的最小HP是
     -dungeon[i][j] + 1.如果格子的值非负，那么最小HP需求就是1.

     这里可以看出DP的方向是从最右下角开始一直到左上角。首先dp[m-1][n-1] = Math.max(1, -dungeon[m-1][n-1] + 1).

     递归方程是dp[i][j] = Math.max(1, Math.min(dp[i+1][j], dp[i][j+1]) - dungeon[i][j]).
     */
    // 重要思想: DP 求右下角的值从左上向右下递归,求左上从右下向左上递归

    public int calculateMinimumHP(int[][] dungeon) {
        if (dungeon == null || dungeon.length == 0 || dungeon[0].length == 0) {
            return 0;
        }
        int m = dungeon.length, n = dungeon[0].length;
        int[][] dp = new int[m][n];
        dp[m - 1][n - 1] = Math.max(1, -dungeon[m - 1][n - 1] + 1);
        for (int i = m - 2; i >= 0; i--) {
            dp[i][n - 1] = Math.max(dp[i + 1][n - 1] - dungeon[i][n - 1], 1);
        }
        for (int i = n - 2; i >= 0; i--) {
            dp[m - 1][i] = Math.max(dp[m - 1][i + 1] - dungeon[m - 1][i], 1);
        }
        for (int i = m - 2; i >= 0; i--) {
            for (int j = n - 2; j >= 0; j--) {
                dp[i][j] = Math.max(Math.min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j], 1);
            }
        }
        return dp[0][0];
    }

    /**
     * Longest Increasing Subsequence: Given an unsorted array of integers, find the length of longest increasing subsequence.

     For example,
     Given [10, 9, 2, 5, 3, 7, 101, 18],
     The longest increasing subsequence is [2, 3, 7, 101], therefore the length is 4. Note that there may be more than one LIS combination, it is only necessary for you to return the length.

     Your algorithm should run in O(n2) complexity.

     Follow up: Could you improve it to O(n log n) time complexity?
     * @param nums
     * @return
     */

    public int lengthOfLIS(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int[] size = new int[nums.length];
        size[0] = 1;
        int result = 1;
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                size[i] = Math.max(size[i], nums[i] > nums[j] ? size[j] + 1 : 1);
            }
            result = Math.max(result, size[i]);
        }
        return result;
    }

    /**
     * Russian Doll Envelopes: Similar to Longest Increasing Subsequence.
     * You have a number of envelopes with widths and heights given as a pair of integers (w, h). One envelope can fit
     * into another if and only if both the width and height of one envelope is greater than the width and height of
     * the other envelope.

     What is the maximum number of envelopes can you Russian doll? (put one inside other)

     Example:
     Given envelopes = [[5,4],[6,4],[6,7],[2,3]], the maximum number of envelopes you can Russian doll is 3 ([2,3] =>
     [5,4] => [6,7]).
     */
    class EnvelopeComparator implements Comparator<int[]> {
        public int compare(int[] e1, int[] e2) {
            if (e1[0] == e2[0]) {
                return e1[1] - e2[1];
            } else {
                return e1[0] - e2[0];
            }
        }
    }
    public class Solution {
        public int maxEnvelopes(int[][] envelopes) {
            if (envelopes == null || envelopes.length == 0) {
                return 0;
            }
            Arrays.sort(envelopes, new EnvelopeComparator());
            int max = 1;
            int[] size = new int[envelopes.length];
            size[0] = 1;
            for (int i = 1; i < envelopes.length; i++) {
                for (int j = 0; j < i; j++) {
                    size[i] = Math.max(size[i], (envelopes[i][0] > envelopes[j][0] && envelopes[i][1] > envelopes[j][1]) ? size[j] + 1 : 1);
                }
                max = Math.max(size[i], max);
            }
            return max;
        }
    }

    /**
     * Climbing stairs: You are climbing a stair case. It takes n steps to reach to the top.
     * Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
     * state f[i]: distinct ways to climb i steps: f[i] = f[i - 1] + f[i - 2], f[1] = 1, f[2] = 2;
     */
    public int climbStairs(int n) {
        if (n <= 3) {
            return n;
        }
        int f1 = 1;
        int f2 = 2;
        int f = 0;
        for (int i = 3; i <= n; i++) {
            f = f1 + f2;
            f1 = f2;
            f2 = f;
        }
        return f;
    }
    /**
     * House Robber: You are a professional robber planning to rob houses along a street. Each house has a certain
     * amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses
     * have security system connected and it will automatically contact the police if two adjacent houses were broken
     * into on the same night.
     * Given a list of non-negative integers representing the amount of money of each house, determine the maximum
     * amount of money you can rob tonight without alerting the police.
     * rob[i]: amount of money when the ith house got robbed
     * norob[i]: amount of money when the ith house does not got robbed
     * rob[i] = norob[i - 1] + nums[i], norob[i] = max(norob[i  - 1], rob[i - 1])
     */
    public int rob(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int[] rob = new int[nums.length];
        int[] norob = new int[nums.length];
        int[] max = new int[nums.length];
        rob[0] = max[0] = nums[0];
        int result = 0;
        for (int i = 1; i < nums.length; i++) {
            rob[i] = norob[i - 1] + nums[i];
            norob[i] = Math.max(norob[i - 1], rob[i - 1]);
            max[i] = Math.max(rob[i], norob[i]);
            result = Math.max(result, max[i]);
        }
        return Math.max(result, max[0]);
    }

    /**
     * Regular Expression Matching: Implement regular expression matching with support for '.' and '*'.
     * '.' Matches any single character.
     * '*' Matches zero or more of the preceding element.
     * The matching should cover the entire input string (not partial).
     * The function prototype should be:
     * bool isMatch(const char *s, const char *p)
     * Some examples:
     * isMatch("aa","a") → false
     * isMatch("aa","aa") → true
     * isMatch("aaa","aa") → false
     * isMatch("aa", "a*") → true
     * isMatch("aa", ".*") → true
     * isMatch("ab", ".*") → true
     * isMatch("aab", "c*a*b") → true
     *
     * isMatch[i][j]: First j characters of p matches first i characters of s:
     * if p.charAt(j - 1) == s.charAt(i - 1) || p.charAt(j-1) == '.'
     * match[i][j] = match[i-1][j-1];
     * if p.charAt(j - 1) == '*'
     * if (s.charAt(i-1) == p.charAt(j-2) || p.charAt(j-2) == '.') {
     // x* zero matched ||  x* match more than or equal to once
     match[i][j] = match[i][j-2] || match[i-1][j];
     } else {
     // x* zero matched, igonored
     match[i][j] = match[i][j-2];
     }
     * initial: isMatch[0][0] = true; isMatch[0][j] = true if for 0 : j - 1, all p.charAt(j - 1) == '*'
     */
    public boolean isMatch(String s, String p) {
        boolean[][] match = new boolean[s.length() + 1][p.length() + 1];
        match[0][0]= true;
        for (int i = 2; i <= p.length(); i++) {
            match[0][i] = p.charAt(i - 1) == '*' ? match[0][i-2] : false;
        }
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 1; j <= p.length(); j++) {
                if (p.charAt(j - 1) == s.charAt(i - 1) || p.charAt(j-1) == '.') {
                    match[i][j] = match[i-1][j-1];
                } else if (p.charAt(j - 1) == '*') {
                    if (s.charAt(i-1) == p.charAt(j-2) || p.charAt(j-2) == '.') {
                        // x* zero matched ||  x* match more than or equal to once
                        match[i][j] = match[i][j-2] || match[i-1][j];
                    } else {
                        // x* zero matched, igonored
                        match[i][j] = match[i][j-2];
                    }
                }
            }
        }
        return match[s.length()][p.length()];
    }

    /**
     * Edit Distance: Given two words word1 and word2, find the minimum number of steps required to convert word1 to word2. (each operation is counted as 1 step.)

     You have the following 3 operations permitted on a word:

     a) Insert a character
     b) Delete a character
     c) Replace a character

     f[i][j]: min distance first i of word1, first j of word2 distance
     if (char i = char j): Math.min(f[i - 1][j - 1], f [i - 1][j] + 1, f[i][j - 1] + 1)
     else Math.min(f [i - 1][j] + 1, f[i][j - 1], f[i - 1][j - 1]) + 1
     initialize: f[0][j] = j, f[i][0] = i;
     */
    public int minDistance(String word1, String word2) {
        if ((word1 == null && word2 == null) || (word1.length() == 0 && word2.length() == 0)) {
            return 0;
        } else if (word1 == null || word1.length() == 0) {
            return word2.length();
        } else if (word2 == null || word2.length() == 0) {
            return word1.length();
        }
        int m = word1.length(), n = word2.length();
        int[][] minDis = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            minDis[i][0] = i;
        }
        for (int j = 0; j <= n; j++) {
            minDis[0][j] = j;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    minDis[i][j] = Math.min(minDis[i - 1][j - 1], Math.min(minDis[i - 1][j], minDis[i][j - 1]) + 1);
                } else {
                    minDis[i][j] = Math.min(Math.min(minDis[i - 1][j], minDis[i][j - 1]), minDis[i - 1][j - 1]) + 1;
                }
            }
        }
        return minDis[m][n];
    }
    /**
     * Best Time to Buy and Sell Stock IV: Say you have an array for which the ith element is the price of a given stock on day i.

     Design an algorithm to find the maximum profit. You may complete at most k transactions.

     Note:
     You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).

     local[i][j]: 在到达第i天时最多可进行j次交易并且最后一次交易在最后一天卖出的最大利润，此为局部最优。
     然后我们定义global[i][j]为在到达第i天时最多可进行j次交易的最大利润，此为全局最优。
     global[i][j] = max(local[i][j], global[i - 1][j]);// last day sell, last day not sell
     local[i][j] = max(global[i - 1][j - 1] + max(diff, 0), local[i - 1][j] + diff)// last day buy and sell , buy before last day, sell last day
     第一个是全局到i-1天进行j-1次交易，然后加上今天的交易，如果今天是赚钱的话（也就是前面只要j-1次交易，最后一次交易取当前天），
     第二个量则是取local第i-1天j次交易，然后加上今天的差值（这里因为local[i-1][j]比如包含第i-1天卖出的交易，所以现在变成第i天卖出，
     并不会增加交易次数，而且这里无论diff是不是大于0都一定要加上，因为否则就不满足local[i][j]必须在最后一天卖出的条件了）
     */
    public int maxProfit(int k, int[] prices) {
        if (prices == null || prices.length <= 1) {
            return 0;
        }
        int max = 0;
        if (k >= prices.length) {
            for (int i = 1; i < prices.length; i++) {
                if (prices[i] - prices[i - 1] > 0) {
                    max += prices[i] - prices[i - 1];
                }
            }
            return max;
        }
        int[][] local = new int[k + 1][prices.length];
        int[][] global = new int[k + 1][prices.length];
        for (int i = 1; i <= k; i++) {
            for (int j = 1; j < prices.length; j++) {
                int diff = prices[j] - prices[j - 1];
                local[i][j] = Math.max(global[i - 1][j - 1] + Math.max(diff, 0), local[i][j - 1] + diff);
                global[i][j] = Math.max(global[i][j - 1], local[i][j]);
            }
        }
        return global[k][prices.length - 1];
    }
    /**
     * Perfect Squares: Given a positive integer n, find the least number of perfect square numbers (for example,
     * 1, 4, 9, 16, ...) which sum to n.

     For example, given n = 12, return 3 because 12 = 4 + 4 + 4; given n = 13, return 2 because 13 = 4 + 9.

     */
    public int numSquares(int n) {
        int[] result = new int[n+1];
        for (int i = 1; i <= n; i++) {
            result[i] = i;
            for (int j = 1; j * j <= i; j++) {
                result[i] = Math.min(result[i - j * j] + 1, result[i]);
            }
        }
        return result[n];
    }

    /**
     * Distinct Subsequences: Given a string S and a string T, count the number of distinct subsequences of T in S.

     A subsequence of a string is a new string which is formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (ie, "ACE" is a subsequence of "ABCDE" while "AEC" is not).

     Here is an example:
     S = "rabbbit", T = "rabbit"

     Return 3.
     f[i][j]: number of distinct subsequences of the first i characters of S of the first j characters of T
     if (char i = char j): f[i][j] = f[i - 1][j] + 1
     else f[i][j] = 0;
     initialize: f[i][0] = 1;
     */
    public int numDistinct(String s, String t) {
        if (s == null || s.length() == 0 || s.length() < t.length()) {
            return 0;
        }
        int m = s.length(), n = t.length();
        int[][] count = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            count[i][0] = 1;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= i && j <= n; j++) {
                if (s.charAt(i - 1) == t.charAt(j - 1)) {
                    count[i][j] = count[i - 1][j - 1] + count[i - 1][j];
                } else {
                    count[i][j] = count[i - 1][j];
                }
            }
        }
        return count[m][n];
    }
    /**
     * Triangle: Given a triangle, find the minimum path sum from top to bottom. Each step you may move to adjacent numbers on the row below.

     For example, given the following triangle
     [
     [2],
     [3,4],
     [6,5,7],
     [4,1,8,3]
     ]
     The minimum path sum from top to bottom is 11 (i.e., 2 + 3 + 5 + 1 = 11).
     */
    public int minimumTotal(List<List<Integer>> triangle) {
        if (triangle == null) return 0;
        // if (triangle.size() == 1) return triangle.get(0).get(0);
        int[] dp = new int[triangle.size()];
        for (int i = 0; i < triangle.get(triangle.size() - 1).size(); i++) {
            dp[i] = triangle.get(triangle.size() - 1).get(i);
        }
        for (int i = triangle.size() - 2; i >= 0; i--) {
            for (int j = 0; j < triangle.get(i).size(); j++) {
                dp[j] = Math.min(dp[j], dp[j+1]) + triangle.get(i).get(j);
            }
        }
        return dp[0];
    }

    /**
     * Burst Balloons: Given n balloons, indexed from 0 to n-1. Each balloon is painted with a number on it represented by array nums. You are asked to burst all the balloons. If the you burst balloon i you will get nums[left] * nums[i] * nums[right] coins. Here left and right are adjacent indices of i. After the burst, the left and right then becomes adjacent.

     Find the maximum coins you can collect by bursting the balloons wisely.

     Note:
     (1) You may imagine nums[-1] = nums[n] = 1. They are not real therefore you can not burst them.
     (2) 0 ≤ n ≤ 500, 0 ≤ nums[i] ≤ 100

     Example:

     Given [3, 1, 5, 8]

     Return 167

     nums = [3,1,5,8] --> [3,5,8] -->   [3,8]   -->  [8]  --> []
     coins =  3*1*5      +  3*5*8    +  1*3*8      + 1*8*1   = 167
     必须先算长度为1的dp[left][right], 然后propagate到更长的长度
     */
    public int maxCoins(int[] nums) {
        int n = nums.length;
        int [][]dp = new int [n+2][n+2];
        int [] arr = new int [n+2];
        for (int i = 1; i <= n; i++){
            arr[i] = nums[i-1];
        }
        arr[0] = 1;
        arr[n+1] = 1;
        for (int len = 1; len <= n; ++len) {
            for (int left = 1; left <= n - len + 1; ++left) {
                int right = left + len - 1;
                for (int k = left; k <= right; ++k) {
                    dp[left][right] = Math.max(dp[left][right], arr[left - 1] * arr[k] * arr[right + 1] + dp[left][k - 1] + dp[k + 1][right]);
                }
            }
        }
        return dp[1][n];
    }

    /**
     * Range Sum Query - Immutable:
     * Given an integer array nums, find the sum of the elements between indices i and j (i ≤ j), inclusive.

     Example:
     Given nums = [-2, 0, 3, -5, 2, -1]

     sumRange(0, 2) -> 1
     sumRange(2, 5) -> -1
     sumRange(0, 5) -> -3
     Note:
     You may assume that the array does not change.
     There are many calls to sumRange function
     * @param args
     */
    class NumArray {
        int[] sum;
        public NumArray(int[] nums) {
            sum = new int[nums.length + 1];
            sum[0] = 0;
            for (int i = 1; i <= nums.length; i++) {
                sum[i] = sum[i - 1] + nums[i - 1];
            }
        }

        public int sumRange(int i, int j) {
            return sum[j + 1] - sum[i];
        }
    }

    /**
     * Unique Path: A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

     The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

     How many possible unique paths are there?
     * @param args
     */
    public int uniquePaths(int m, int n) {
        int[][] paths = new int[m][n];
        if (m == 0 || n == 0) return 0;
        if (m == 1 || n == 1) return 1;
        for (int i = 0; i < n; i++) {
            paths[0][i] = 1;
        }
        for (int i = 0; i < m; i++) {
            paths[i][0] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                paths[i][j] = paths[i-1][j] + paths[i][j-1];
            }
        }
        return paths[m-1][n-1];
    }

    public static void main(String args[]) {
        int[] num = {3, 1, 5, 8};
        OneDimensionDP dp = new OneDimensionDP();
        System.out.println(dp.maxCoins(num));
    }


























































}
