import java.util.Arrays;
import java.util.Comparator;

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
}
