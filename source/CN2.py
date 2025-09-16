import pandas as pd
import numpy as np
import copy
import collections as clc
import time

class CN2:

    def __init__(self, train_data_csv, test_data_csv):
        self.train_set = pd.read_csv(train_data_csv)
        self.test_set = pd.read_csv(test_data_csv)
        self.min_significance = 0.001  
        self.max_star_size = 6         # Increased to allow more complex rules
        self.max_iterations = 50       

    def fit_CN2(self):
        print("Starting CN2 model training...")
        start_time = time.time()
        selectors = self.find_attribute_pairs()
        remaining_examples = self.train_set
        rule_list = []
        iteration = 0

        while len(remaining_examples) >= 1:
            iteration += 1
            if iteration > self.max_iterations:
                print("Maximum iterations reached, stopping")
                break
            print(f"Iteration {iteration}: {len(remaining_examples)} examples remaining")
            best_new_rule_significance = 1
            rules_to_specialise = []
            existing_results = pd.DataFrame()

            while best_new_rule_significance > self.min_significance:
                if len(rules_to_specialise) == 0:
                    ordered_rule_results = self.apply_and_order_rules_by_score(selectors, remaining_examples)
                    trimmed_rule_results = ordered_rule_results[0:self.max_star_size]
                else:
                    specialised_rules = self.specialise_complex(rules_to_specialise, selectors)
                    ordered_rule_results = self.apply_and_order_rules_by_score(specialised_rules, remaining_examples)
                    trimmed_rule_results = ordered_rule_results[0:self.max_star_size]

                existing_results = pd.concat([existing_results, trimmed_rule_results], ignore_index=True)
                existing_results = self.order_rules(existing_results).iloc[0:2]
                rules_to_specialise = trimmed_rule_results['rule']
                best_new_rule_significance = trimmed_rule_results['significance'].values[0]
                print(f"Best significance: {best_new_rule_significance:.2f}")

            if not existing_results.empty:
                best_rule = (existing_results['rule'].iloc[0], existing_results['predict_class'].iloc[0],
                             existing_results['num_insts_covered'].iloc[0])
                best_rule_coverage_index, best_rule_coverage_df = self.complex_coverage(best_rule[0], remaining_examples)
                rule_list.append(best_rule)
                remaining_examples = remaining_examples.drop(best_rule_coverage_index)
                print(f"Rule added: {best_rule}")
            else:
                print("No rule found, stopping")
                break

        print(f"Training completed. Generated {len(rule_list)} rules")
        print(f"Training time: {time.time() - start_time:.2f} seconds")
        return rule_list

    def test_fitted_model(self, rule_list, data_set='default'):
        if type(data_set) == str:
            data_set = self.test_set

        default_class = self.train_set['class'].mode()[0]
        predictions = pd.Series(index=data_set.index, dtype=object)
        remaining_examples = data_set.copy()

        for rule in rule_list:
            rule_coverage_indexes, rule_coverage_dataframe = self.complex_coverage(rule[0], remaining_examples)
            if len(rule_coverage_dataframe) > 0:
                predicted_class = rule[1]
                predictions.loc[rule_coverage_indexes] = predicted_class
                remaining_examples = remaining_examples.drop(rule_coverage_indexes)

        if not remaining_examples.empty:
            predictions.loc[remaining_examples.index] = default_class

        y_true = data_set['class'].values
        y_pred = predictions.values

        cm, labels = self.confusion_matrix(y_true, y_pred)
        metrics = self.calculate_metrics(cm, labels)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")

        list_of_row_dicts = []
        for rule in rule_list:
            rule_coverage_indexes, rule_coverage_dataframe = self.complex_coverage(rule[0], data_set)
            if len(rule_coverage_dataframe) == 0:
                row_dictionary = {
                    'rule': rule, 'pred_class': 'zero coverage', 'rule_acc': 0,
                    'num_examples': 0, 'num_correct': 0, 'num_wrong': 0
                }
            else:
                class_of_covered_examples = rule_coverage_dataframe['class']
                class_counts = class_of_covered_examples.value_counts()
                rule_accuracy = class_counts.values[0] / sum(class_counts)
                num_correct = class_counts.values[0]
                num_wrong = sum(class_counts.values) - num_correct
                row_dictionary = {
                    'rule': rule, 'pred_class': rule[1], 'rule_acc': rule_accuracy,
                    'num_examples': len(rule_coverage_indexes), 'num_correct': num_correct,
                    'num_wrong': num_wrong
                }
            list_of_row_dicts.append(row_dictionary)

        results = pd.DataFrame(list_of_row_dicts)
        return results, metrics

    def confusion_matrix(self, y_true, y_pred):
        """Manually compute the confusion matrix."""
        labels = np.unique(np.concatenate((y_true, y_pred)))
        n_classes = len(labels)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        
        for true, pred in zip(y_true, y_pred):
            true_idx = label_to_idx[true]
            pred_idx = label_to_idx[pred]
            cm[true_idx, pred_idx] += 1
        
        return cm, labels

    def calculate_metrics(self, cm, labels):
        """Calculate accuracy, macro precision, recall, and F1-score from confusion matrix."""
        n_classes = len(labels)
        TP = np.diag(cm)  # True Positives
        FP = cm.sum(axis=0) - TP  # False Positives
        FN = cm.sum(axis=1) - TP  # False Negatives

        # Accuracy
        total = cm.sum()
        accuracy = TP.sum() / total if total > 0 else 0

        # Per-class precision, recall, and F1
        precision = np.zeros(n_classes)
        recall = np.zeros(n_classes)
        f1 = np.zeros(n_classes)

        for i in range(n_classes):
            precision[i] = TP[i] / (TP[i] + FP[i]) if (TP[i] + FP[i]) > 0 else 0
            recall[i] = TP[i] / (TP[i] + FN[i]) if (TP[i] + FN[i]) > 0 else 0
            f1[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

        # Macro-averaged metrics
        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1 = f1.mean()

        return {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1
        }

    def apply_and_order_rules_by_score(self, complexes, data_set='default'):
        if type(data_set) == str:
            data_set = self.train_set
        list_of_row_dicts = []
        for row in complexes:
            rule_coverage = self.complex_coverage(row, data_set)[1]
            rule_length = len(row)
            if len(rule_coverage) == 0:
                row_dictionary = {
                    'rule': row, 'predict_class': 'dud rule',
                    'entropy': 10, 'laplace_accuracy': 0,
                    'significance': 0, 'length': rule_length,
                    'num_insts_covered': 0, 'specificity': 0
                }
            else:
                num_examples_covered = len(rule_coverage)
                entropy_of_rule = self.rule_entropy(rule_coverage)
                significance_of_rule = self.rule_significance(rule_coverage)
                laplace_accuracy_of_rule = self.rule_laplace_accuracy(rule_coverage)
                class_attrib = rule_coverage['class']
                class_counts = class_attrib.value_counts()
                majority_class = class_counts.axes[0][0]
                rule_specificity = class_counts.values[0] / sum(class_counts)
                row_dictionary = {
                    'rule': row, 'predict_class': majority_class,
                    'entropy': entropy_of_rule, 'laplace_accuracy': laplace_accuracy_of_rule,
                    'significance': significance_of_rule, 'length': rule_length,
                    'num_insts_covered': num_examples_covered, 'specificity': rule_specificity
                }
            list_of_row_dicts.append(row_dictionary)

        rules_and_stats = pd.DataFrame(list_of_row_dicts)
        ordered_rules_and_stats = self.order_rules(rules_and_stats)
        return ordered_rules_and_stats

    def order_rules(self, dataFrame_of_rules):
        ordered_rules_and_stats = dataFrame_of_rules.sort_values(
            ['entropy', 'length', 'num_insts_covered'], ascending=[True, True, False])
        ordered_rules_and_stats = ordered_rules_and_stats.reset_index(drop=True)
        return ordered_rules_and_stats

    def find_attribute_pairs(self):
        attributes = self.train_set.columns.values.tolist()
        del attributes[-1]
        possAttribVals = {}
        for att in attributes:
            possAttribVals[att] = set(self.train_set[att])

        attrib_value_pairs = []
        for key in possAttribVals.keys():
            for possVal in possAttribVals[key]:
                attrib_value_pairs.append([(key, possVal)])
        return attrib_value_pairs

    def specialise_complex(self, target_complexes, selectors):
        provisional_specialisations = []
        max_combinations = 1000
        for targ_complex in target_complexes:
            for selector in selectors:
                if len(provisional_specialisations) > max_combinations:
                    print("Maximum combinations reached, stopping")
                    break
                if type(targ_complex) == tuple:
                    comp_to_specialise = [copy.copy(targ_complex)]
                else:
                    comp_to_specialise = copy.copy(targ_complex)

                comp_to_specialise.append(selector[0])
                count_of_selectors_in_complex = clc.Counter(comp_to_specialise)
                flag = True
                for count in count_of_selectors_in_complex.values():
                    if count > 1:
                        flag = False

                if flag:
                    provisional_specialisations.append(comp_to_specialise)

        return provisional_specialisations

    def build_rule(self, passed_complex):
        atts_used_in_rule = [selector[0] for selector in passed_complex]
        set_of_atts_used_in_rule = set(atts_used_in_rule)

        if len(set_of_atts_used_in_rule) < len(atts_used_in_rule):
            return False

        rule = {}
        attributes = self.train_set.columns.values.tolist()
        for att in attributes:
            rule[att] = list(set(self.train_set[att]))

        for att_val_pair in passed_complex:
            att = att_val_pair[0]
            val = att_val_pair[1]
            rule[att] = [val]
        return rule

    def complex_coverage(self, passed_complex, data_set='default'):
        if type(data_set) == str:
            data_set = self.train_set
        
        if not passed_complex:  # If the rule is empty
            return [], pd.DataFrame()
        
        mask = pd.Series(True, index=data_set.index)
        for attr, val in passed_complex:
            if attr in data_set.columns:
                mask &= (data_set[attr] == val)
            else:
                return [], pd.DataFrame()
        
        covered_data = data_set[mask]
        return covered_data.index.tolist(), covered_data

    def check_rule_datapoint(self, datapoint, complex):
        if type(complex) == tuple:
            if datapoint[complex[0]] == complex[1]:
                return True
            else:
                return False
        if type(complex) == list:
            result = True
            for selector in complex:
                if datapoint[selector[0]] != selector[1]:
                    result = False
            return result

    def rule_entropy(self, covered_data):
        class_series = covered_data['class']
        num_instances = len(class_series)
        class_counts = class_series.value_counts()
        class_probabilities = class_counts / num_instances
        log2_of_classprobs = np.log2(class_probabilities)
        plog2p = class_probabilities * log2_of_classprobs
        entropy = -plog2p.sum()
        return entropy

    def rule_significance(self, covered_data):
        covered_classes = covered_data['class']
        covered_num_instances = len(covered_classes)
        covered_counts = covered_classes.value_counts()
        covered_probs = covered_counts / covered_num_instances

        train_classes = self.train_set['class']
        train_num_instances = len(train_classes)
        train_counts = train_classes.value_counts()
        train_probs = train_counts / train_num_instances

        significance = 2 * (covered_probs * np.log(covered_probs / train_probs)).sum()
        return significance

    def rule_laplace_accuracy(self, covered_data):
        class_series = covered_data['class']
        class_counts = class_series.value_counts()
        num_instances = len(class_series)
        num_classes = len(class_counts)
        num_pred_class = class_counts.iloc[0]
        laplace_accuracy = (num_pred_class + 1) / (num_instances + num_classes)
        return laplace_accuracy

if __name__ == '__main__':
    lenseFit = CN2('train_set_lense.csv', 'test_set_lense.csv')
    lenseRules = lenseFit.fit_CN2()
    lenseTest, metrics = lenseFit.test_fitted_model(lenseRules, lenseFit.test_set)
    lenseTest.to_csv('lense_test_results.csv')