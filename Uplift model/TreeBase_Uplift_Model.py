import pandas as pd
import numpy as np
from causalml.inference.tree import UpliftTreeClassifier,uplift_tree_plot,DecisionTree,CausalTreeRegressor
from collections import defaultdict
import pydotplus
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.tree._export import _DOTTreeExporter, _color_brew
from io import StringIO
from sklearn.utils.validation import check_is_fitted
from sklearn.tree import _tree
from joblib import Parallel, delayed
import scipy.sparse as sp


# -------------------- UpliftTree -------------------- 

# to model continuous outcome variables, inherit UpliftTreeClassifier and refactor some functions
class uplift_tree(UpliftTreeClassifier):
    def __init__(self, control_name='0', max_features=None, max_depth=3, min_samples_leaf=100,
                 min_samples_treatment=10, n_reg=100,evaluationFunction='KL',
                 normalization=True, random_state=None,is_y_clssifier=True,
                 honesty=False,estimation_sample_size=0.5):
        super().__init__(control_name=control_name, max_features=max_features, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                 min_samples_treatment=min_samples_treatment, n_reg=n_reg, 
                 normalization=normalization, random_state=random_state,
                 honesty=honesty,estimation_sample_size=estimation_sample_size)
        if evaluationFunction == 'KL':
            self.evaluationFunction = self.evaluate_KL_new
        elif evaluationFunction == 'ED':
            self.evaluationFunction = self.evaluate_ED
        elif evaluationFunction == 'Chi':
            self.evaluationFunction = self.evaluate_Chi
        elif evaluationFunction == 'DDP':
            self.evaluationFunction = self.evaluate_DDP
        elif evaluationFunction == 'CTS':
            self.evaluationFunction = self.evaluate_CTS
        elif evaluationFunction == 'CAPE':
            self.evaluationFunction = self.evaluate_CAPE
        else:
            self.evaluationFunction = self.evaluate_KL_new
        self.fitted_uplift_tree = None
        
        self.is_y_clssifier = is_y_clssifier


    def group_uniqueCounts(self, treatment_idx, y):
        results = []
        for i in range(self.n_class):
            filt = treatment_idx == i
            n_pos = len(y[filt])
            sum_pos = y[filt].sum()
            var_pos=np.var(y[filt])

            # [sum(Y, T = i), N(T = i)]
            results.append([sum_pos, n_pos, var_pos])

        return results


    def tree_node_summary(self, treatment_idx, y, min_samples_treatment=10, n_reg=100, parentNodeSummary=None):
        # counts: [SUM(Y, T=0), N(T=0)], [SUM(Y, T=1), N(T=1)], ...]
        counts = self.group_uniqueCounts(treatment_idx, y)

        cav_yt=np.cov(np.array([treatment_idx,y]).astype(float))[0,1]
        var_t=np.var(treatment_idx)

        # nodeSummary: [[P(Y|T=0), N(T=0), Var(Y|T=0), sum(Y|T=0)], [P(Y|T=1), N(T=1), Var(Y|T=1), sum(Y|T=1)], ...]
        nodeSummary = []
        # Iterate the control and treatment groups
        for i, count in enumerate(counts):
            n_pos = count[0]
            n = count[1]
            if parentNodeSummary is None:
                p = n_pos / n if n > 0 else 0.
            elif n > min_samples_treatment:
                p = (n_pos + parentNodeSummary[i][0] * n_reg) / (n + n_reg)
            else:
                p = parentNodeSummary[i][0]

            nodeSummary.append([p, n, n_pos, count[2]])

        nodeSummary.append([cav_yt/var_t])

        # print(nodeSummary)
        # print(' ')
        return nodeSummary


    def uplift_classification_results(self, treatment_idx, y):
        # counts: [SUM(Y, T=0), N(T=0)], [SUM(Y, T=1), N(T=1)], ...]
        counts = self.group_uniqueCounts(treatment_idx, y)

        # real result：[P(Y, T=0), P(Y, T=0), ...]
        res = []
        for count in counts:
            n_pos = count[0]
            n = count[1]
            p = n_pos / n if n > 0 else 0.
            res.append(p)

        return res


    @staticmethod
    def evaluate_KL_new(nodeSummary):
        '''
        Calculate KL Divergence as split evaluation criterion for a given node.
        Args
        ----
        nodeSummary : list of list
            The tree node summary statistics, [P(Y=1|T), N(T)], produced by tree_node_summary()
            method.
        Returns
        -------
        d_res : KL Divergence
        '''
        nodeSummary_new=nodeSummary[:-1]
        return UpliftTreeClassifier.evaluate_KL(nodeSummary_new)

    @staticmethod
    def evaluate_ED(nodeSummary):
        '''
        Calculate Euclidean Distance as split evaluation criterion for a given node.
        Args
        ----
        nodeSummary : dictionary
            The tree node summary statistics, produced by tree_node_summary()
            method.
        Returns
        -------
        d_res : Euclidean Distance
        '''
        pc = nodeSummary[0][0]
        d_res = 0
        for treatment_group in nodeSummary[1:-1]:
            d_res += 2*(treatment_group[0] - pc)**2
        return d_res

    @staticmethod
    def evaluate_Chi(nodeSummary):
        '''
        Calculate Chi-Square statistic as split evaluation criterion for a given node.
        Args
        ----
        nodeSummary : dictionary
            The tree node summary statistics, produced by tree_node_summary() method.
        Returns
        -------
        d_res : Chi-Square
        '''
        pc = nodeSummary[0][0]
        d_res = 0
        for treatment_group in nodeSummary[1:-1]:
            d_res += ((treatment_group[0] - pc) ** 2 / max(0.1 ** 6, pc)
                      + (treatment_group[0] - pc) ** 2 / max(0.1 ** 6, 1 - pc))
        return d_res

    @staticmethod
    def evaluate_DDP(nodeSummary):
        '''
        Calculate Delta P as split evaluation criterion for a given node.
        Args
        ----
        nodeSummary : list of list
            The tree node summary statistics, [P(Y=1|T), N(T)], produced by tree_node_summary() method.
        Returns
        -------
        d_res : Delta P
        '''
        pc = nodeSummary[0][0]
        d_res = 0
        for treatment_group in nodeSummary[1:-1]:
            d_res += treatment_group[0] - pc
        return d_res

    @staticmethod
    def evaluate_CTS(nodeSummary):
        '''
        Calculate CTS (conditional treatment selection) as split evaluation criterion for a given node.
        Args
        ----
        nodeSummary : list of list
            The tree node summary statistics, [P(Y=1|T), N(T)], produced by tree_node_summary() method.
        Returns
        -------
        d_res : Chi-Square
        '''
        return -max([stat[0] for stat in nodeSummary[:-1]])

    @staticmethod
    def evaluate_CAPE(nodeSummary):
        return nodeSummary[-1][0]


    def growDecisionTreeFrom(self, X, treatment_idx, y, X_val, treatment_val_idx, y_val,
                             early_stopping_eval_diff_scale=1, max_depth=10,
                             min_samples_leaf=100, depth=1,
                             min_samples_treatment=10, n_reg=100,
                             parentNodeSummary=None):

        if len(X) == 0:
            return DecisionTree(classes_=self.classes_)

        # Current node summary: [P(Y=1|T), N(T)]
        currentNodeSummary = self.tree_node_summary(treatment_idx, y,
                                                    min_samples_treatment=min_samples_treatment,
                                                    n_reg=n_reg,
                                                    parentNodeSummary=parentNodeSummary)
        currentScore = self.evaluationFunction(currentNodeSummary)

        # Prune Stats
        maxAbsDiff = 0
        if self.is_y_clssifier==True:
            maxDiff = -1.
        else:
            maxDiff = -999.
        bestTreatment = 0       # treatment index for the control group
        suboptTreatment = 0     # treatment index for the control group
        maxDiffTreatment = 0    # treatment index for the control group
        maxDiffSign = 0
        bestDiff = 0
        upliftScore = []
        p_value = [0]
        diff_train=''
        p_c, n_c, sum_c, var_c = currentNodeSummary[0]
        for i_tr in range(1, self.n_class):
            p_t, n_t, sum_t, var_t = currentNodeSummary[i_tr]
            # P(Y|T=t) - P(Y|T=0)
            diff = p_t - p_c
            diff_train=diff_train+str(round(diff,4))+', '

            if self.is_y_clssifier==True:
                p_value.append((1. - stats.norm.cdf(abs(p_c - p_t) / np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c))) * 2)
            else:
                p_value.append((1. - stats.norm.cdf(abs(p_c - p_t) / np.sqrt(var_c/n_c + var_t/n_t))) * 2)

            if abs(diff) >= maxAbsDiff:
                maxDiffTreatment = i_tr
                maxDiffSign = np.sign(diff)
                maxAbsDiff = abs(diff)
            if diff >= maxDiff:
                maxDiff = diff
                suboptTreatment = i_tr
                if diff > 0:
                    bestTreatment = i_tr
                    bestDiff = diff

        upliftScore = [[bestDiff, p_value[bestTreatment], bestTreatment],[maxDiff,p_value[suboptTreatment], suboptTreatment]]

        bestGain = 0.0
        bestGainImp = 0.0
        bestAttribute = None

        # last column is the result/target column, 2nd to the last is the treatment group
        columnCount = X.shape[1]
        if (self.max_features and self.max_features > 0 and self.max_features <= columnCount):
            max_features = self.max_features
        else:
            max_features = columnCount

        for col in list(self.random_state_.choice(a=range(columnCount), size=max_features, replace=False)):
            columnValues = X[:, col]
            # unique values
            lsUnique = np.unique(columnValues)

            if np.issubdtype(lsUnique.dtype, np.number):
                if len(lsUnique) > 10:
                    lspercentile = np.percentile(columnValues, [3, 5, 10, 20, 30, 50, 40, 60, 70, 80, 90, 95, 97])
                    lsUnique = np.unique(lspercentile)
                # else:
                #     lspercentile = np.percentile(lsUnique, [10, 30, 50, 70, 90])


            for value in lsUnique:
                X_l, X_r, w_l, w_r, y_l, y_r = self.divideSet(X, treatment_idx, y, col, value)
                # check the split validity on min_samples_leaf  372
                if (len(X_l) < min_samples_leaf or len(X_r) < min_samples_leaf):
                    continue
                # summarize notes
                # Gain -- Entropy or Gini
                p = float(len(X_l)) / len(X)
                leftNodeSummary = self.tree_node_summary(w_l, y_l,
                                                         min_samples_treatment=min_samples_treatment,
                                                         n_reg=n_reg,
                                                         parentNodeSummary=currentNodeSummary)

                rightNodeSummary = self.tree_node_summary(w_r, y_r,
                                                          min_samples_treatment=min_samples_treatment,
                                                          n_reg=n_reg,
                                                          parentNodeSummary=currentNodeSummary)

                # check the split validity on min_samples_treatment
                assert len(leftNodeSummary) == len(rightNodeSummary)
                
                if X_val is not None:
                    X_val_l, X_val_r, w_val_l, w_val_r, y_val_l, y_val_r = self.divideSet(X_val, treatment_val_idx, y_val, col, value)
                    leftNodeSummary_val = self.tree_node_summary(w_val_l, y_val_l,
                                                             parentNodeSummary=currentNodeSummary)
                    rightNodeSummary_val = self.tree_node_summary(w_val_r, y_val_r,
                                                              parentNodeSummary=currentNodeSummary)
                    early_stopping_flag = False
                    for k in range(len(leftNodeSummary_val)):
                        if ((abs(leftNodeSummary_val[k][0]-leftNodeSummary[k][0]) > 
                             min(leftNodeSummary_val[k][0],leftNodeSummary[k][0])/early_stopping_eval_diff_scale)  or
                         (abs(rightNodeSummary_val[k][0]-rightNodeSummary[k][0]) > 
                          min(rightNodeSummary_val[k][0],rightNodeSummary[k][0])/early_stopping_eval_diff_scale)):
                            early_stopping_flag = True
                            break
                    if early_stopping_flag:
                        continue

                node_mst = min([stat[1] for stat in leftNodeSummary[:-1] + rightNodeSummary[:-1]])
                if node_mst < min_samples_treatment:
                    continue

                # evaluate the split
                if self.evaluationFunction == self.evaluate_CTS:
                    leftScore1 = self.evaluationFunction(leftNodeSummary)
                    rightScore2 = self.evaluationFunction(rightNodeSummary)
                    gain = (currentScore - p * leftScore1 - (1 - p) * rightScore2)
                    gain_for_imp = (len(X) * currentScore - len(X_l) * leftScore1 - len(X_r) * rightScore2)
                elif self.evaluationFunction == self.evaluate_DDP:
                    leftScore1 = self.evaluationFunction(leftNodeSummary)
                    rightScore2 = self.evaluationFunction(rightNodeSummary)
                    gain = np.abs(leftScore1 - rightScore2)
                    gain_for_imp = np.abs(len(X_l) * leftScore1 - len(X_r) * rightScore2)
                elif self.evaluationFunction == self.evaluate_CAPE:
                    leftScore1 = self.evaluationFunction(leftNodeSummary)
                    rightScore2 = self.evaluationFunction(rightNodeSummary)
                    gain = len(X_l)*len(X_r)/(len(X)*len(X))*((leftScore1 - rightScore2)**2)
                    gain_for_imp = (len(X_l) * leftScore1 + len(X_r) * rightScore2 - len(X) * currentScore)
                else:
                    leftScore1 = self.evaluationFunction(leftNodeSummary)
                    rightScore2 = self.evaluationFunction(rightNodeSummary)
                    gain = (p * leftScore1 + (1 - p) * rightScore2 - currentScore)
                    gain_for_imp = (len(X_l) * leftScore1 + len(X_r) * rightScore2 - len(X) * currentScore)
                    if self.normalization:
                        n_c = currentNodeSummary[0][1]
                        n_c_left = leftNodeSummary[0][1]
                        n_t = [tr[1] for tr in currentNodeSummary[1:-1]]
                        n_t_left = [tr[1] for tr in leftNodeSummary[1:-1]]

                        norm_factor = self.normI(n_c, n_c_left, n_t, n_t_left, alpha=0.9)
                    else:
                        norm_factor = 1
                    gain = gain / norm_factor
                if (gain > bestGain and len(X_l) > min_samples_leaf and len(X_r) > min_samples_leaf):
                    bestGain = gain
                    bestGainImp = gain_for_imp
                    bestAttribute = (col, value)
                    best_set_left = [X_l, w_l, y_l, None, None, None]
                    best_set_right = [X_r, w_r, y_r, None, None, None]
                    if X_val is not None:
                        best_set_left = [X_l, w_l, y_l, X_val_l, w_val_l, y_val_l]
                        best_set_right = [X_r, w_r, y_r, X_val_r, w_val_r, y_val_r]

        dcY = {'impurity': '%.3f' % currentScore, 'samples': '%d' % len(X)}
        # Add treatment size
        dcY['group_size'] = ''
        for i, summary in enumerate(currentNodeSummary[:-1]):
            dcY['group_size'] += ' ' + self.classes_[i] + ': ' + str(summary[1])
        dcY['upliftScore'] = [[round(upliftScore[0][0], 4), round(upliftScore[0][1], 4), upliftScore[0][2]],
                              [round(upliftScore[1][0], 4), round(upliftScore[1][1], 4), upliftScore[1][2]]]
        dcY['matchScore'] = [round(upliftScore[0][0], 4),round(upliftScore[1][0], 4)]
        dcY['p_value']=p_value
        dcY['diff_train']=diff_train[:-2]
        dcY['diff_test']=diff_train[:-2]

        if bestGain > 0 and depth < max_depth:
            self.feature_imp_dict[bestAttribute[0]] += bestGainImp
            trueBranch = self.growDecisionTreeFrom(
                *best_set_left, self.early_stopping_eval_diff_scale, 
                max_depth, min_samples_leaf,
                depth + 1, min_samples_treatment=min_samples_treatment,
                n_reg=n_reg, parentNodeSummary=currentNodeSummary
            )
            falseBranch = self.growDecisionTreeFrom(
                *best_set_right, self.early_stopping_eval_diff_scale,
                max_depth, min_samples_leaf,
                depth + 1, min_samples_treatment=min_samples_treatment,
                n_reg=n_reg, parentNodeSummary=currentNodeSummary
            )

            return DecisionTree(
                classes_=self.classes_,
                col=bestAttribute[0], value=bestAttribute[1],
                trueBranch=trueBranch, falseBranch=falseBranch, summary=dcY,
                maxDiffTreatment=maxDiffTreatment, maxDiffSign=maxDiffSign,
                nodeSummary=currentNodeSummary,
                backupResults=self.uplift_classification_results(treatment_idx, y),
                bestTreatment=bestTreatment, upliftScore=upliftScore
            )
        else:
            if self.evaluationFunction == self.evaluate_CTS:
                return DecisionTree(
                    classes_=self.classes_,
                    results=self.uplift_classification_results(treatment_idx, y),
                    summary=dcY, nodeSummary=currentNodeSummary,
                    bestTreatment=bestTreatment, upliftScore=upliftScore
                )
            else:
                return DecisionTree(
                    classes_=self.classes_,
                    results=self.uplift_classification_results(treatment_idx, y),
                    summary=dcY, maxDiffTreatment=maxDiffTreatment,
                    maxDiffSign=maxDiffSign, nodeSummary=currentNodeSummary,
                    bestTreatment=bestTreatment, upliftScore=upliftScore
                )


    def fillTree(self, X, treatment_idx, y, tree):
        """ Fill the data into an existing tree.
        This is a lower-level function to execute on the tree filling task.
        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.
        treatment_idx : array-like, shape = [num_samples]
            An array containing the treatment group index for each unit.
        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
        tree : object
            object of DecisionTree class
        Returns
        -------
        self : object
        """
        # Current Node Summary for Validation Data Set
        currentNodeSummary = self.tree_node_summary(treatment_idx, y,
                                                    min_samples_treatment=0,
                                                    n_reg=0,
                                                    parentNodeSummary=None)
        tree.nodeSummary = currentNodeSummary

        # Divide sets for child nodes
        if tree.trueBranch or tree.falseBranch:
            X_l, X_r, w_l, w_r, y_l, y_r = self.divideSet(X, treatment_idx, y, tree.col, tree.value)

            # recursive call for each branch
            if tree.trueBranch is not None:
                self.fillTree(X_l, w_l, y_l, tree.trueBranch)
            if tree.falseBranch is not None:
                self.fillTree(X_r, w_r, y_r, tree.falseBranch)

        # Update Information

        # test_cate
        diff_test=''
        p_c= currentNodeSummary[0][0]
        for i in range(1,len(currentNodeSummary)-1):
            diff_test=diff_test+str(round(currentNodeSummary[i][0]-p_c,4))+', '
        tree.summary['diff_test']=diff_test[:-2]

        # matchScore
        matchScore = [round(currentNodeSummary[tree.bestTreatment][0] - currentNodeSummary[0][0],4),
                      round(currentNodeSummary[tree.upliftScore[1][2]][0] - currentNodeSummary[0][0],4)]
        tree.matchScore = matchScore
        tree.summary['matchScore'] = matchScore

        # Samples, Group_size
        tree.summary['samples'] = len(y)
        tree.summary['group_size'] = ''
        for treatment_group, summary in zip(self.classes_, currentNodeSummary[:-1]):
            tree.summary['group_size'] += ' ' + treatment_group + ': ' + str(summary[1])
        # classProb
        if tree.results is not None:
            tree.results = self.uplift_classification_results(treatment_idx, y)
        return self


# add bestTreatment
def uplift_tree_plot(decisionTree, x_names):
    """
    Convert the tree to dot graph for plots.
    Args
    ----
    decisionTree : object
        object of DecisionTree class
    x_names : list
        List of feature names
    Returns
    -------
    Dot class representing the tree graph.
    """

    # Column Heading
    dcHeadings = {}
    for i, szY in enumerate(x_names + ["treatment_group_key"]):
        szCol = "Column %d" % i
        dcHeadings[szCol] = str(szY)

    dcNodes = defaultdict(list)
    """Plots the obtained decision tree. """

    def toString(
        iSplit,
        decisionTree,
        bBranch,
        szParent="null",
        indent="",
        indexParent=0,
        upliftScores=list(),
    ):
        if decisionTree.results is not None:  # leaf node
            lsY = []
            for tr, p in zip(decisionTree.classes_, decisionTree.results):
                lsY.append(f"{tr}:{p:.2f}")
            dcY = {"name": ", ".join(lsY), "parent": szParent}
            dcSummary = decisionTree.summary
            # 改用suboptTreatment的uplift值来画图
            upliftScores += [dcSummary["matchScore"][1]]
            p_value_list=''
            for p in dcSummary["p_value"][1:]:
                p_value_list=p_value_list+str(round(p,4))+', '
            dcNodes[iSplit].append(
                [
                    "leaf",
                    dcY["name"],
                    szParent,
                    bBranch,
                    str(-round(float(decisionTree.summary["impurity"]), 3)),
                    dcSummary["samples"],
                    dcSummary["group_size"],
                    p_value_list[:-2],
                    dcSummary["diff_train"],
                    dcSummary["diff_test"],
                    dcSummary["upliftScore"],
                    dcSummary["matchScore"],
                    decisionTree.bestTreatment,
                    indexParent,
                ]
            )
        else:
            szCol = "Column %s" % decisionTree.col
            if szCol in dcHeadings:
                szCol = dcHeadings[szCol]
            if isinstance(decisionTree.value, int) or isinstance(
                decisionTree.value, float
            ):
                decision = "%s >= %s" % (szCol, decisionTree.value)
            else:
                decision = "%s == %s" % (szCol, decisionTree.value)

            indexOfLevel = len(dcNodes[iSplit])
            toString(
                iSplit + 1,
                decisionTree.trueBranch,
                True,
                decision,
                indent + "\t\t",
                indexOfLevel,
                upliftScores,
            )
            toString(
                iSplit + 1,
                decisionTree.falseBranch,
                False,
                decision,
                indent + "\t\t",
                indexOfLevel,
                upliftScores,
            )
            dcSummary = decisionTree.summary
            # 改用suboptTreatment的uplift值来画图
            upliftScores += [dcSummary["matchScore"][1]]
            p_value_list=''
            for p in dcSummary["p_value"][1:]:
                p_value_list=p_value_list+str(round(p,4))+' , '
            dcNodes[iSplit].append(
                [
                    iSplit + 1,
                    decision,
                    szParent,
                    bBranch,
                    str(-round(float(decisionTree.summary["impurity"]), 3)),
                    dcSummary["samples"],
                    dcSummary["group_size"],
                    p_value_list[:-3],
                    dcSummary["diff_train"],
                    dcSummary["diff_test"],
                    dcSummary["upliftScore"],
                    dcSummary["matchScore"],
                    decisionTree.bestTreatment,
                    indexParent,
                ]
            )
    upliftScores = list()
    toString(0, decisionTree, None, upliftScores=upliftScores)

    upliftScoreToColor = dict()
    try:
        # calculate colors for nodes based on uplifts
        minUplift = min(upliftScores)
        maxUplift = max(upliftScores)
        upliftLevels = [
            (uplift - minUplift) / (maxUplift - minUplift) for uplift in upliftScores
        ]  # min max scaler
        # 改用suboptTreatment的uplift值来画图
        baseUplift = float(decisionTree.summary.get("matchScore")[1])
        baseUpliftLevel = (baseUplift - minUplift) / (
            maxUplift - minUplift
        )  # min max scaler normalization
        white = np.array([255.0, 255.0, 255.0])
        blue = np.array([31.0, 119.0, 180.0])
        green = np.array([0.0, 128.0, 0.0])
        for i, upliftLevel in enumerate(upliftLevels):
            if upliftLevel >= 0:  # go blue
                color = upliftLevel * blue + (1 - upliftLevel) * white
            else:  # go green
                color = (1 - upliftLevel) * green + upliftLevel * white
            color = [int(c) for c in color]
            upliftScoreToColor[upliftScores[i]] = ("#%2x%2x%2x" % tuple(color)).replace(
                " ", "0"
            )  # color code
    except Exception as e:
        print(e)

    lsDot = [
        "digraph Tree {",
        'node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;',
        "edge [fontname=helvetica] ;",
    ]
    i_node = 0
    dcParent = {}
    totalSample = int(
        decisionTree.summary.get("samples")
    )  # initialize the value with the total sample size at root
    for nSplit in range(len(dcNodes.items())):
        lsY = dcNodes[nSplit]
        indexOfLevel = 0
        for lsX in lsY:
            (
                iSplit,
                decision,
                szParent,
                bBranch,
                szImpurity,
                szSamples,
                szGroup,
                p_value,
                diff_train,
                diff_test,
                upliftScore,
                matchScore,
                bestTreatment,
                indexParent,
            ) = lsX

            sampleProportion = round(int(szSamples) * 100.0 / totalSample, 1)
            if type(iSplit) is int:
                szSplit = "%d-%d" % (iSplit, indexOfLevel)
                dcParent[szSplit] = i_node
                if len(decisionTree.classes_)>2:
                    lsDot.append(
                        "%d [label=<%s<br/> impurity %s<br/> total_sample %s (%s&#37;)<br/> "
                        "p_value : %s <br/> "
                        "uplift_train : %s <br/> "
                        "uplift_test : %s <br/> "
                        'bestTreatment : %s, suboptTreatment : %s >, fillcolor="%s"] ;'
                        % (
                            i_node,
                            decision.replace(">=", "&ge;").replace("?", ""),
                            szImpurity,
                            szSamples,
                            str(sampleProportion),
                            # szGroup,
                            p_value,
                            diff_train,
                            diff_test,
                            str(upliftScore[0][2]),
                            str(upliftScore[1][2]),
                            upliftScoreToColor.get(matchScore[1], "#e5813900"),
                        )
                    )
                else:
                    lsDot.append(
                        "%d [label=<%s<br/> impurity %s<br/> total_sample %s (%s&#37;)<br/> "
                        "p_value : %s <br/> "
                        "uplift_train : %s <br/> "
                        'uplift_test : %s >, fillcolor="%s"] ;'
                        % (
                            i_node,
                            decision.replace(">=", "&ge;").replace("?", ""),
                            szImpurity,
                            szSamples,
                            str(sampleProportion),
                            # szGroup,
                            p_value,
                            diff_train,
                            diff_test,
                            upliftScoreToColor.get(matchScore[1], "#e5813900"),
                        )
                    )
            else:
                if len(decisionTree.classes_)>2:
                    lsDot.append(
                        "%d [label=< impurity %s<br/> total_sample %s (%s&#37;)<br/> "
                        "p_value : %s <br/> "
                        "uplift_train : %s <br/> "
                        "uplift_test : %s <br/> "
                        "bestTreatment : %s, suboptTreatment : %s <br/> "
                        'mean %s >, fillcolor="%s"] ;'
                        % (
                            i_node,
                            szImpurity,
                            szSamples,
                            str(sampleProportion),
                            # szGroup,
                            p_value,
                            diff_train,
                            diff_test,
                            str(upliftScore[0][2]),
                            str(upliftScore[1][2]),
                            decision,
                            upliftScoreToColor.get(matchScore[1], "#e5813900"),
                        )
                    )
                else:
                    lsDot.append(
                        "%d [label=< impurity %s<br/> total_sample %s (%s&#37;)<br/> "
                        "p_value : %s <br/> "
                        "uplift_train : %s <br/> "
                        "uplift_test : %s <br/> "
                        'mean %s >, fillcolor="%s"] ;'
                        % (
                            i_node,
                            szImpurity,
                            szSamples,
                            str(sampleProportion),
                            # szGroup,
                            p_value,
                            diff_train,
                            diff_test,
                            decision,
                            upliftScoreToColor.get(matchScore[1], "#e5813900"),
                        )
                    )

            if szParent != "null":
                if bBranch:
                    szAngle = "45"
                    szHeadLabel = "True"
                else:
                    szAngle = "-45"
                    szHeadLabel = "False"
                szSplit = "%d-%d" % (nSplit, indexParent)
                p_node = dcParent[szSplit]
                if nSplit == 1:
                    lsDot.append(
                        '%d -> %d [labeldistance=2.5, labelangle=%s, headlabel="%s"] ;'
                        % (p_node, i_node, szAngle, szHeadLabel)
                    )
                else:
                    lsDot.append("%d -> %d ;" % (p_node, i_node))
            i_node += 1
            indexOfLevel += 1
    lsDot.append("}")
    dot_data = "\n".join(lsDot)
    graph = pydotplus.graph_from_dot_data(dot_data)
    return graph


# add other information of leaf nodes, such as cate, p value, sample size
def uplift_tree_string(decisionTree, x_names):
    """
    Convert the tree to string for print.
    Args
    ----
    decisionTree : object
        object of DecisionTree class
    x_names : list
        List of feature names
    Returns
    -------
    A string representation of the tree.
    """
    # Column Heading
    dcHeadings = {}
    for i, szY in enumerate(x_names + ["treatment_group_key"]):
        szCol = "Column %d" % i
        dcHeadings[szCol] = str(szY)
    def toString(decisionTree, path, df_leaf):
        if decisionTree.results is not None:  #leaf node

            p_value_list=''
            for p in decisionTree.summary["p_value"][1:]:
                p_value_list=p_value_list+str(round(p,4))+', '

            p_c, n_c, sum_c = decisionTree.nodeSummary[0][0:3]
            sum_t=''
            n_t=''
            cate=''
            for i in range(1,len(decisionTree.nodeSummary)-1):
                sum_t = sum_t + str(decisionTree.nodeSummary[i][2])+', '
                n_t = n_t + str(decisionTree.nodeSummary[i][1])+', '
                cate = cate + str(round(decisionTree.nodeSummary[i][0]-p_c,4))+', '

            df_leaf.loc[len(df_leaf)] = [path[:-4],decisionTree.upliftScore[0][2],
                                         decisionTree.upliftScore[1][2],p_value_list[:-2],cate[:-2],n_t[:-2],n_c,sum_t[:-2],sum_c]

        else:
            szCol = "Column %s" % decisionTree.col
            if szCol in dcHeadings:
                szCol = dcHeadings[szCol]
            if isinstance(decisionTree.value, int) or isinstance(decisionTree.value, float):
                decision = "%s >= %s" % (szCol, decisionTree.value)
                decision_no = "%s < %s" % (szCol, decisionTree.value)
            else:
                decision = "%s == %s?" % (szCol, decisionTree.value)
                decision_no = "%s != %s" % (szCol, decisionTree.value)
            toString(decisionTree.trueBranch, path + decision + " and ", df_leaf)
            toString(decisionTree.falseBranch, path + decision_no + " and ", df_leaf)

    df_leaf=pd.DataFrame(columns=['path','bestTreatment','suboptTreatment','p_value','uplift','num_t','num_c','sum_t','sum_c'])
    toString(decisionTree, '', df_leaf)

    return df_leaf





def UpliftTree(df,t,y,x,train=None,is_y_clssifier=True,max_depth=5,criterion='KL',
               min_split_per=0.01,graph_out=True,imp_out=True,honesty=False,estimation_sample_size=0.5):

    """
    df : dataframe 
        containing treatment,outcome and confounding variables.

    t : str
        the column name of treatment.

    y : str
        the column name of outcome.

    x : list of str
        the column name of confounding variables.

    s_y_clssifier : bool, optional (default=True)
        whether the outcome variable is categorical.

    max_depth : int, optional (default=3)
        the maximum depth of the tree.

    criterions : string
        choose from one of the models: 'KL', 'ED', 'Chi', 'CTS', 'DDP'.

    min_split_per : float,optional (default=0.01)
        the minimum proportion of samples required to be split at a leaf node.

    """

    # divide training set and test set
    train_col=train
    if train_col==None:
        np.random.seed(25)
        msk = np.random.rand(len(df)) < 0.8
        df[train_col]=1
        df.loc[~msk,train_col]=0

    df_train=df.loc[df[train_col]==1].reset_index(drop=True)
    df_test=df.loc[df[train_col]==0].reset_index(drop=True)
    del df


    # modeling
    uplift_model = uplift_tree(is_y_clssifier=is_y_clssifier,
                               max_depth=max_depth, min_samples_leaf=int(len(df_train)*min_split_per),n_reg=100,evaluationFunction=criterion,
                               honesty=honesty,estimation_sample_size=estimation_sample_size)
    uplift_model.fit(X=df_train[x].values,treatment=df_train[t].astype('str').values, y=df_train[y].values)
    df_leaf_train=uplift_tree_string(uplift_model.fitted_uplift_tree,x)

    uplift_model.fill(X=df_test[x].values, treatment=df_test[t].astype('str').values, y=df_test[y].values)
    df_leaf_test=uplift_tree_string(uplift_model.fitted_uplift_tree,x)

    # info of tree leaves
    df_leaf=pd.DataFrame()
    if uplift_model.n_class>2:
        df_leaf[['path','bestTreatment','suboptTreatment','p_value',
                 'uplift_train']]=df_leaf_train[['path','bestTreatment','suboptTreatment','p_value','uplift']]
        df_leaf[['uplift_test','num_t_test','num_c_test',
                 'sum_t_test','sum_c_test']]=df_leaf_test[['uplift','num_t','num_c','sum_t','sum_c']]
        # df_leaf['num_c_test_per']=df_leaf['num_c_test']/sum(df_leaf['num_c_test'])
    else:
        df_leaf[['path','p_value','uplift_train']]=df_leaf_train[['path','p_value','uplift']]
        df_leaf[['uplift_test','num_t_test','num_c_test','sum_t_test','sum_c_test']]=df_leaf_test[['uplift','num_t','num_c','sum_t','sum_c']]


    if (graph_out==True)&(imp_out==True):
        graph = uplift_tree_plot(uplift_model.fitted_uplift_tree, x)
        x_imp = pd.Series(uplift_model.feature_importances_, index=x)
        return graph,x_imp,df_leaf
    elif (graph_out==True)&(imp_out==False):
        graph = uplift_tree_plot(uplift_model.fitted_uplift_tree, x)
        return graph,df_leaf
    elif (graph_out==False)&(imp_out==True):
        x_imp = pd.Series(uplift_model.feature_importances_, index=x)
        return x_imp,df_leaf
    else:
        return df_leaf



# -------------------- CausalTree -------------------- 

# plot the causal tree
class Sentinel:
    def __repr__(self):
        return '"tree.dot"'

SENTINEL = Sentinel()

class _DOTCTreeExporter(_DOTTreeExporter):
    def __init__(
        self,
        causal_tree: CausalTreeRegressor,
        groups_count: bool,
        treatment_groups: tuple,
        nodes_group_cnt: dict,

        out_file=SENTINEL,
        max_depth=None,
        feature_names=None,
        class_names=None,
        label="all",
        filled=False,
        leaves_parallel=False,
        impurity=True,
        node_ids=False,
        proportion=False,
        rotate=False,
        rounded=False,
        special_characters=False,
        precision=3,
        fontname="helvetica",
    ):
        super().__init__(
            out_file=out_file,
            max_depth=max_depth,
            feature_names=feature_names,
            class_names=class_names,
            label=label,
            filled=filled,
            leaves_parallel=leaves_parallel,
            impurity=impurity,
            node_ids=node_ids,
            proportion=proportion,
            rotate=rotate,
            rounded=rounded,
            special_characters=special_characters,
            precision=precision,
            fontname=fontname,
        )
        self.causal_tree = causal_tree
        self.groups_count = groups_count
        self.treatment_groups = treatment_groups
        self.nodes_group_cnt = nodes_group_cnt
        # self.n_features_in_ = causal_tree.n_features_in_

    def node_to_str(
        self, tree: _tree.Tree, node_id: int, criterion: str or object
    ) -> str:
        """
        Generate the node content string
        Args:
            tree:      Tree class
            node_id:   int, Tree node id
            criterion: str or object, split criterion
        Returns: str, node content
        """

        # Should labels be shown?
        labels = (self.label == "root" and node_id == 0) or self.label == "all"

        characters = self.characters
        node_string = characters[-1]


        # Write decision criteria
        if tree.children_left[node_id] != _tree.TREE_LEAF:
            # Always write node decision criteria, except for leaves
            if self.feature_names is not None:
                feature = self.feature_names[tree.feature[node_id]]
            else:
                feature = "X%s%s%s" % (
                    characters[1],
                    tree.feature[node_id],
                    characters[2],
                )
            node_string += "%s %s %s%s" % (
                feature,
                characters[3],
                round(tree.threshold[node_id], self.precision),
                characters[4],
            )

        # Write impurity
        if self.impurity:
            if not isinstance(criterion, str):
                criterion = "impurity"
            if labels:
                node_string += "%s = " % criterion
            node_string += (
                    str(round(tree.impurity[node_id], self.precision)) + characters[4]
                )

        # p_value
        node_string += ("p_value"+" = ")
        node_string += (
                str(round(self.nodes_group_cnt[node_id]['p_value'], self.precision)) + characters[4]
            )            

        # Write node sample count
        if labels:
            node_string += "samples_test = "
            if self.proportion:
                percent = 100.0 * (self.nodes_group_cnt[node_id]['test_per'][-1])
                node_string += str(round(percent, 1)) + "%" + characters[4]
            else:
                node_string += str(self.nodes_group_cnt[node_id]['test_num'][-1])+ characters[4]

        for str_t in ('train','test'):
            value =self.nodes_group_cnt[node_id]['uplift_'+str_t]
            if self.proportion and tree.n_classes[0] != 1:
                # For classification this will show the proportion of samples
                value = value / tree.weighted_n_node_samples[node_id]
            if labels:
                node_string += ("uplift_"+str_t+" = ")
            if tree.n_classes[0] == 1:
                # Regression
                value_text = np.around(value, self.precision)
            elif self.proportion:
                # Classification
                value_text = np.around(value, self.precision)
            elif np.all(np.equal(np.mod(value, 1), 0)):
                # Classification without floating-point weights
                value_text = value.astype(int)
            else:
                # Classification with floating-point weights
                value_text = np.around(value, self.precision)
            # Strip whitespace
            value_text = str(value_text.astype("S32")).replace("b'", "'")
            value_text = value_text.replace("' '", ", ").replace("'", "")
            if tree.n_classes[0] == 1 and tree.n_outputs == 1:
                value_text = value_text.replace("[", "").replace("]", "")
            value_text = value_text.replace("\n ", characters[4])
            node_string += value_text + characters[4]

            # print(node_string + characters[5])
        return node_string + characters[5]


    def get_color(self, value: np.ndarray) -> str:
        """
        Compute HTML color for a Tree node
        Args:
            value: Tree node value
        Returns: str, html color code in #RRGGBB format
        """
        # Regression tree or multi-output
        white = np.array([255.0, 255.0, 255.0])
        blue = np.array([31.0, 119.0, 180.0])
        green = np.array([0.0, 128.0, 0.0])
        alpha = float(value - self.colors["bounds"][0]) / (self.colors["bounds"][1] - self.colors["bounds"][0])

        alpha = 0 if np.isnan(alpha) else alpha
        # Compute the color as alpha against white
        if value>0:
            color = alpha * blue + (1 - alpha) * white
        else:
            color = alpha * white + (1 - alpha) * green
        color = [int(c) for c in color]
        return ("#%2x%2x%2x" % tuple(color)).replace(" ", "0")



    def get_fill_color(self, tree: _tree.Tree, node_id: int) -> str:
        """
         Fetch appropriate color for node
        Args:
            tree:    Tree class
            node_id: int, node index
        Returns: str
        """
        if "rgb" not in self.colors:
            # Initialize colors and bounds if required
            self.colors["rgb"] = _color_brew(tree.n_classes[0])
            if tree.n_outputs != 1:
                # Find max and min impurities for multi-output
                self.colors["bounds"] = (
                    np.nanmin(-tree.impurity),
                    np.nanmax(-tree.impurity),
                )
            elif tree.n_classes[0] == 1 and len(np.unique(tree.value)) != 1:
                # Find max and min values in leaf nodes for regression
                self.colors["bounds"] = (np.nanmin([self.nodes_group_cnt[ni]['uplift_test'] for ni in range(len(tree.value))]), 
                                         np.nanmax([self.nodes_group_cnt[ni]['uplift_test'] for ni in range(len(tree.value))]))
        if tree.n_outputs == 1:
            node_val = self.nodes_group_cnt[node_id]['uplift_test'] / (self.nodes_group_cnt[node_id]['test_num'][0]+
                                                                       self.nodes_group_cnt[node_id]['test_num'][0])
            if tree.n_classes[0] == 1:
                # Regression
                node_val = self.nodes_group_cnt[node_id]['uplift_test']
        else:
            # If multi-output color node by impurity
            node_val = -tree.impurity[node_id]
        return self.get_color(node_val)


    def export(self, decision_tree):
        # Check length of feature_names before getting into the tree node
        # Raise error if length of feature_names does not match
        # n_features_in_ in the decision_tree
        if self.feature_names is not None:
            if len(self.feature_names) != self.causal_tree.n_features_in_:
                raise ValueError(
                    "Length of feature_names, %d does not match number of features, %d"
                    % (len(self.feature_names), decision_tree.n_features_in_)
                )
        # each part writes to out_file
        self.head()
        # Now recurse the tree and add node & edge attributes
        if isinstance(decision_tree, _tree.Tree):
            self.recurse(decision_tree, 0, criterion="impurity")
        else:
            self.recurse(decision_tree.tree_, 0, criterion=decision_tree.criterion)

        self.tail()



def plot_causal_tree(
    causal_tree: CausalTreeRegressor,
    *,
    groups_count: bool = True,
    treatment_groups: tuple = (0, 1),
    nodes_group_cnt: dict = {},
    out_file=SENTINEL,

    max_depth: int = None,
    feature_names: list = None,
    class_names: list = None,
    label: str = "all",
    filled: bool = False,
    leaves_parallel: bool = False,
    impurity: bool = True,
    node_ids: bool = False,
    proportion=False,
    rotate: bool = False,
    rounded: bool = False,
    special_characters: bool = False,
    precision=3,
    fontname="helvetica",
):
    check_is_fitted(causal_tree)
    own_file = False
    return_string = False
    try:
        if isinstance(out_file, str):
            out_file = open(out_file, "w", encoding="utf-8")
            own_file = True

        if out_file is None:
            return_string = True
            out_file = StringIO()

        exporter = _DOTCTreeExporter(
            causal_tree=causal_tree,
            groups_count=groups_count,
            treatment_groups=treatment_groups,
            nodes_group_cnt=nodes_group_cnt,
            out_file=out_file,
            max_depth=max_depth,
            feature_names=feature_names,
            class_names=class_names,
            label=label,
            filled=filled,
            leaves_parallel=leaves_parallel,
            impurity=impurity,
            node_ids=node_ids,
            proportion=proportion,
            rotate=rotate,
            rounded=rounded,
            special_characters=special_characters,
            precision=precision,
            fontname=fontname,
        )

        dot_data = exporter.export(causal_tree)
        if return_string:
            return exporter.out_file.getvalue()

    finally:
        if own_file:
            out_file.close()


                
def CausalTree(df,t,y,x,train=None,is_y_clssifier=True,max_depth=5,min_split_per=0.01,graph_out=True):
    
    # divide training set and test set
    train_col=train
    if train_col==None:
        np.random.seed(25)
        msk = np.random.rand(len(df)) < 0.8
        df[train_col]=1
        df.loc[~msk,train_col]=0
        
    df_train=df.loc[df[train_col]==1].reset_index(drop=True)
    df_test=df.loc[df[train_col]==0].reset_index(drop=True)
    del df
    
    ate_train=df_train.loc[df_train[t]==1,y].mean()-df_train.loc[df_train[t]==0,y].mean()
    ate_test=df_test.loc[df_test[t]==1,y].mean()-df_test.loc[df_test[t]==0,y].mean()    
    
    # modeling
    ct=CausalTreeRegressor(control_name=0, max_depth=max_depth, min_samples_leaf=int(len(df_train)*min_split_per), random_state=None,groups_cnt=True)
    # split the nodes with the data from the training data
    ct.fit(X=df_train[x], treatment=df_train[t], y=df_train[y])

    # Compute causal effects of nodes using data from the training data
    nodes_ids = np.array(range(ct.tree_.node_count))
    leaves_ids = nodes_ids[ct.is_leaves]

    df_train['leaf_id']=ct.tree_.apply(df_train[x].values.astype(np.float32))
    df_train['num']=[1]*len(df_train)

    df_train_result=pd.DataFrame(leaves_ids,columns=['leaf_id'])
    df_train_result['t_sum']=(df_train.loc[(df_train[t]>0),['leaf_id',y]].groupby(['leaf_id']).sum().reset_index(drop=True))[y]
    df_train_result['c_sum']=(df_train.loc[(df_train[t]==0),['leaf_id',y]].groupby(['leaf_id']).sum().reset_index(drop=True))[y]
    df_train_result['t_num']=(df_train.loc[(df_train[t]>0),['leaf_id','num']].groupby(['leaf_id']).count().reset_index(drop=True))['num']
    df_train_result['c_num']=(df_train.loc[(df_train[t]==0),['leaf_id','num']].groupby(['leaf_id']).count().reset_index(drop=True))['num']
    df_train_result['t_var']=(df_train.loc[(df_train[t]>0),['leaf_id',y]].groupby(['leaf_id']).var().reset_index(drop=True))[y]
    df_train_result['c_var']=(df_train.loc[(df_train[t]==0),['leaf_id',y]].groupby(['leaf_id']).var().reset_index(drop=True))[y]

    p_c=df_train_result['c_sum']/df_train_result['c_num']
    p_t=df_train_result['t_sum']/df_train_result['t_num']
    n_c=df_train_result['c_num']
    n_t=df_train_result['t_num']
    var_c=df_train_result['c_var']
    var_t=df_train_result['t_var']
    if is_y_clssifier==True:
        df_train_result['p_value'] = (1. - stats.norm.cdf(abs(p_c - p_t) / np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c))) * 2
    else:
        df_train_result['p_value'] = (1. - stats.norm.cdf(abs(p_c - p_t) / np.sqrt(var_c/n_c + var_t/n_t))) * 2



    # Compute causal effects of nodes using data from the testing data
    df_test_leaves_id=ct.tree_.apply(df_test[x].values.astype(np.float32))
    df_test['leaf_id']=pd.Series(sp.csr_matrix(df_test_leaves_id).toarray().ravel(), dtype=int)
    df_test['num']=[1]*len(df_test)

    df_test_result=df_train_result.loc[:,['leaf_id','p_value']]
    df_test_result['t_sum']=(df_test.loc[(df_test[t]>0)&(df_test[y]>0),['leaf_id',y]].groupby(['leaf_id']).sum().reset_index(drop=True))[y]
    df_test_result['c_sum']=(df_test.loc[(df_test[t]==0)&(df_test[y]>0),['leaf_id',y]].groupby(['leaf_id']).sum().reset_index(drop=True))[y]
    df_test_result['t_num']=(df_test.loc[(df_test[t]>0),['leaf_id','num']].groupby(['leaf_id']).count().reset_index(drop=True))['num']
    df_test_result['c_num']=(df_test.loc[(df_test[t]==0),['leaf_id','num']].groupby(['leaf_id']).count().reset_index(drop=True))['num']
    df_test_result['uplift_test']=df_test_result['t_sum']/df_test_result['t_num']-df_test_result['c_sum']/df_test_result['c_num']
    df_test_result=df_test_result.sort_values(by='uplift_test',ascending=False).reset_index(drop=True)
    df_test_result['ate_train']=[ate_train]*len(df_test_result)
    df_test_result['ate_test']=[ate_test]*len(df_test_result)

    # print the path information of the node
    tree=ct.tree_
    feature_names=x
    node_indicator = tree.decision_path(df_train[x].values.astype(np.float32))
    # df_train_leaf_id = tree.apply(df_train[x].values.astype(np.float32))
    parent = {}

    for i in range(tree.node_count):
        if tree.children_left[i] != -1:
            parent[tree.children_left[i]] = i
        if tree.children_right[i] != -1:
            parent[tree.children_right[i]] = i

    df_test_result['path']=['']*len(df_test_result)
    for leaf in leaves_ids:
        node_index = np.where(df_train['leaf_id'] == leaf)[0]
        str_when=''
        path_printed = False
        for sample_id in node_index:
            path = []
            node_id = node_indicator.indices[node_indicator.indptr[sample_id]:
                                                node_indicator.indptr[sample_id + 1]][-1]
            while node_id != 0:
                path.append(node_id)
                node_id = parent[node_id]
            path.reverse()
            if not path_printed: # 只在第一次输出该叶子节点路径信息
                for i, node in enumerate(path):
                    if tree.children_left[parent[node]] == node:
                        str_when=str_when+("{} <= {:.4f}".format(feature_names[tree.feature[parent[node]]], tree.threshold[parent[node]]))+' and '
                    else:
                        str_when=str_when+("{} > {:.4f}".format(feature_names[tree.feature[parent[node]]], tree.threshold[parent[node]]))+' and '
                # print()
                path_printed = True
        df_test_result.loc[df_test_result['leaf_id']==leaf,'path']=str_when[:len(str_when)-5]


    # calculate node information
    nodes_group_cnt={}
    for i in range(tree.node_count):
        nodes_group_cnt[i]={}
        for k in ['train_num','test_num','test_sum']:
            nodes_group_cnt[i][k]={0:0,1:0}

        nodes_group_cnt[i]['df_train']=pd.DataFrame(columns=[t,y,'num'])

    df_test_result['uplift_train']=[0]*len(df_test_result)
    for leaf_i in leaves_ids:
        nodes_group_cnt[leaf_i]['train_num']=ct._groups_cnt[leaf_i]
        nodes_group_cnt[leaf_i]['test_num'][0]=df_test_result.loc[df_test_result['leaf_id']==leaf_i,'c_num'].values[0]
        nodes_group_cnt[leaf_i]['test_num'][1]=df_test_result.loc[df_test_result['leaf_id']==leaf_i,'t_num'].values[0]
        nodes_group_cnt[leaf_i]['test_num'][-1]=nodes_group_cnt[leaf_i]['test_num'][0]+nodes_group_cnt[leaf_i]['test_num'][1]
        nodes_group_cnt[leaf_i]['test_sum'][0]=df_test_result.loc[df_test_result['leaf_id']==leaf_i,'c_sum'].values[0]
        nodes_group_cnt[leaf_i]['test_sum'][1]=df_test_result.loc[df_test_result['leaf_id']==leaf_i,'t_sum'].values[0]
        nodes_group_cnt[leaf_i]['df_train']=df_train.loc[df_train['leaf_id']==leaf_i,[t,y,'num']]
        df_test_result.loc[df_test_result['leaf_id']==leaf_i,'uplift_train']=tree.value[leaf_i][0][0]

        parent_i=parent[leaf_i]
        for k in ['train_num','test_num','test_sum']:
            nodes_group_cnt[parent_i][k][0]+=nodes_group_cnt[leaf_i][k][0]
            nodes_group_cnt[parent_i][k][1]+=nodes_group_cnt[leaf_i][k][1]

        nodes_group_cnt[parent_i]['df_train']=nodes_group_cnt[leaf_i]['df_train'].append(df_train.loc[df_train['leaf_id']==leaf_i,[t,y,'num']])

        while parent_i>0:
            child_i=parent_i
            parent_i=parent[child_i]
            for k in ['train_num','test_num','test_sum']:
                nodes_group_cnt[parent_i][k][0]+=nodes_group_cnt[leaf_i][k][0]
                nodes_group_cnt[parent_i][k][1]+=nodes_group_cnt[leaf_i][k][1]

                nodes_group_cnt[parent_i]['df_train']=nodes_group_cnt[leaf_i]['df_train'].append(df_train.loc[df_train['leaf_id']==leaf_i,[t,y,'num']])
    for i in range(tree.node_count):   
        nodes_group_cnt[i]['train_per']={}
        nodes_group_cnt[i]['train_per'][0]=nodes_group_cnt[i]['train_num'][0]/nodes_group_cnt[0]['train_num'][0]
        nodes_group_cnt[i]['train_per'][1]=nodes_group_cnt[i]['train_num'][1]/nodes_group_cnt[0]['train_num'][1]
        nodes_group_cnt[i]['test_per']={}
        nodes_group_cnt[i]['test_per'][-1]=(nodes_group_cnt[i]['test_num'][0]+nodes_group_cnt[i]['test_num'][1])/(nodes_group_cnt[0]['test_num'][0]+
                                                                                                                  nodes_group_cnt[0]['test_num'][1])
        nodes_group_cnt[i]['test_per'][0]=nodes_group_cnt[i]['test_num'][0]/nodes_group_cnt[0]['test_num'][0]
        nodes_group_cnt[i]['test_per'][1]=nodes_group_cnt[i]['test_num'][1]/nodes_group_cnt[0]['test_num'][1]
        nodes_group_cnt[i]['uplift_train']=tree.value[i][0][0]
        nodes_group_cnt[i]['uplift_test']=(nodes_group_cnt[i]['test_sum'][1]/nodes_group_cnt[i]['test_num'][1]-
                                           nodes_group_cnt[i]['test_sum'][0]/nodes_group_cnt[i]['test_num'][0])

        df_train_leaf=nodes_group_cnt[i]['df_train']
        df_train_leaf=df_train_leaf.reset_index(drop=True)
        t_sum=df_train_leaf.loc[(df_train_leaf[t]>0),y].sum()
        c_sum=df_train_leaf.loc[(df_train_leaf[t]==0),y].sum()
        t_num=df_train_leaf.loc[(df_train_leaf[t]>0),'num'].count()
        c_num=df_train_leaf.loc[(df_train_leaf[t]==0),'num'].count()
        t_var=df_train_leaf.loc[(df_train_leaf[t]>0),y].var()
        c_var=df_train_leaf.loc[(df_train_leaf[t]==0),y].var()
        p_c=c_sum/c_num
        p_t=t_sum/t_num
        n_c=c_num
        n_t=t_num
        var_c=c_var
        var_t=t_var
        if is_y_clssifier==True:
            nodes_group_cnt[i]['p_value'] = (1. - stats.norm.cdf(abs(p_c - p_t) / np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c))) * 2
        else:
            nodes_group_cnt[i]['p_value'] = (1. - stats.norm.cdf(abs(p_c - p_t) / np.sqrt(var_c/n_c + var_t/n_t))) * 2

        nodes_group_cnt[i]['uplift_test']=(nodes_group_cnt[i]['test_sum'][1]/nodes_group_cnt[i]['test_num'][1]-
                                           nodes_group_cnt[i]['test_sum'][0]/nodes_group_cnt[i]['test_num'][0])


    df_leaf=df_test_result.loc[:,['path','p_value','uplift_train','uplift_test','t_sum','c_sum','t_num','c_num']]

    df_leaf.columns=['path','p_value','uplift_train','uplift_test','sum_t_test','sum_c_test','num_t_test','num_c_test']


    if graph_out==True:
        dot_data=plot_causal_tree(
            causal_tree=ct,
            groups_count=True,
            treatment_groups=(0, 1),
            nodes_group_cnt=nodes_group_cnt,
            out_file=None,

            max_depth=max_depth,
            feature_names=x,
            class_names=True,
            label= "all",
            filled=True,
            leaves_parallel=False,
            impurity=True,
            node_ids=False,
            proportion=True,
            rotate=False,
            rounded=False,
            special_characters=False,
            precision=4,
            fontname="helvetica"
        )
        graph = pydotplus.graph_from_dot_data(dot_data)
        return graph,df_leaf
    else:
        return df_leaf