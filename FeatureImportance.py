importances = clf.feature_importances_

# make importance relative to the max importance
feature_importance = 100.0 * (importances / importances.max())
sorted_idx = np.argsort(feature_importance)
feature_names = list(processed_train_features.columns.values)
print(sorted_idx.shape)
print(len(feature_names))
feature_names_sort = [feature_names[indice] for indice in sorted_idx]
pos = np.arange(sorted_idx.shape[0]) + .5
print('Top 25 features are: ')
for feature in feature_names_sort[::-1][:25]:
    print(feature)

#print(feature_names_sort[::-1][:25])
# plot the result
plt.figure(figsize=(10, 45))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, feature_names_sort)
plt.title('Relative Feature Importance', fontsize=20)
plt.show()