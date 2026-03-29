import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


#Loading Data
BeerData = pd.read_csv("beer_profile_and_ratings.csv")

#Cleaning useless data 
columns_to_drop = [
    "Name",
    "Beer Name (Full)",
    "Brewery",
    "Description",
    "review_aroma",
    "review_appearance",
    "review_palate",
    "review_taste",
    "number_of_reviews"
]

BeerData = BeerData.drop(columns=columns_to_drop)

#Verification
#We check the dataset structure before continuing the analysis
print("Dataset overview :")
print(BeerData.head(), "\n")

print("Dataset Dim:")
print(BeerData.shape, "\n")

print("Column type :")
print(BeerData.dtypes, "\n")

print("Missing values :")
print(BeerData.isna().sum(), "\n")

#Group def for comparison 
#We split the dataset into good beers and bad beers to compare their characteristics
top_beers =BeerData[BeerData["review_overall"] >= 4.2].copy() #We decide a good beer is rated 4.2 or more
bad_beers = BeerData[BeerData["review_overall"] <= 3.0].copy() #We decide a bad beer is rated 3.0 or less

print("Number of good beers:",len(top_beers))
print("Number of bad beers  :", len(bad_beers),"\n")

#We stop the code if one category is empty because comparison would be impossible
if len(top_beers) == 0 or len(bad_beers)==0:
    raise ValueError(
        "Un des groupes est vide. Ajuste les seuils (par ex. >= 4.0 et <= 3.2)."
    )


#Analysis of average profiles
#We compute the average profile of good beers and bad beers without the target column
top_profile= top_beers.drop(columns=["review_overall"]).mean(numeric_only=True)
bad_profile= bad_beers.drop(columns=["review_overall"]).mean(numeric_only=True)

#We calculate the average difference between good and bad beers
profile_diff =top_profile-bad_profile

print("Average profiles of good beers : ")
print(top_profile.sort_values(ascending=False), "\n")

print("Average profiles of bad beers : ")
print(bad_profile.sort_values(ascending=False), "\n")

print("Diff (good - bad)")
print(profile_diff.sort_values(ascending=False), "\n")


# Comparison Table
#We gather all descriptive statistics into one comparison table
comparison_BeerData = pd.DataFrame({
    "Top Beers Mean": top_profile,
    "Bad Beers Mean": bad_profile,
    "Difference": profile_diff
}).sort_values(by="Difference", ascending=False)

#print("Comparison Table :")
#print(comparison_BeerData, "\n")


#Targeted analysis of a few important variables
#We keep only the most relevant descriptive variables for the first graph
important_features=[
    "Min IBU",
    "Max IBU",
    "Body",
    "Bitter",
    "Sweet",
    "Sour",
    "Fruits",
    "Hoppy",
    "Spices",
    "Malty"
] #We have removed ABV, Alcohol, Astignancy and Salty wich have the difference between a top tier beer and a bad tier beer


available_features =[col for col in important_features if col in comparison_BeerData.index]

#We sort the selected features by descending difference between good and bad beers
focused_comparison = comparison_BeerData.loc[available_features].sort_values(
    by="Difference", ascending=False
)

#print("\n Key features : Bad beer vs Good Beer")
#print(focused_comparison, "\n")


# First Graphe
#This graph shows which key features differ the most between good and bad beers
plt.figure(figsize=(10,6))
focused_comparison["Difference"].sort_values().plot(kind="barh")
plt.title("Key Feature Differences: Top vs Low Rated Beers")
plt.xlabel("Mean Difference")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

#Preparation for the Model 
#We prepare features and target for machine learning
BeerData_model = BeerData.copy()

X=BeerData_model.drop("review_overall", axis=1)
y=BeerData_model["review_overall"]

le =LabelEncoder()
X["Style"] = le.fit_transform(X["Style"])

#The dataset is split into training and testing sets using an 80/20 ratio.
#This allows the model to learn on the majority of the data while keeping a portion unseen for evaluation.
#A fixed random_state is used to ensure reproducibility of the results. In this case we used 42. 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Model training
#We use Random Forest because it handles non linear relations and feature interactions well
model =RandomForestRegressor(n_estimators=100, random_state=42) #We have choosed 100 as estimators number beacause less than that will not be precise and more doesnt improve that much our model 
model.fit(X_train, y_train)

#We predict beer ratings on the test set
y_prediction = model.predict(X_test)

#We evaluate the model with MSE and R²
# For r2 closer to 1 means better performance
# For mse,  lower MSE means better predictions
mse = mean_squared_error(y_test, y_prediction)
r2 = r2_score(y_test, y_prediction)

print("\n Model Results : ")
print("MSE:", mse)
print("R² :", r2)


#Feature importance
#We extract feature importance from the trained Random Forest model
importance= model.feature_importances_
features= X.columns

features_imp= pd.Series(importance, index=features).sort_values(ascending=False)

#print("\n Feature importance :")
#print(features_imp, "\n")

#This graph shows which variables have the strongest impact on predicted beer ratings
plt.figure(figsize=(10, 6))
features_imp.sort_values().plot(kind="barh")
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()


# 15. Real vs prédit
#This graph compares actual ratings and predicted ratings to evaluate model quality
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_prediction, alpha=0.6)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Beer Ratings")

#The dashed diagonal line represents perfect predictions
minimum_value= min(y_test.min(), y_prediction.min())
maximum_value= max(y_test.max(), y_prediction.max())
plt.plot([minimum_value, maximum_value], [minimum_value, maximum_value], linestyle="--")

plt.tight_layout()
plt.show()

#Error distribution
#This histogram shows how prediction errors are distributed around zero
errors= y_test - y_prediction

plt.figure(figsize=(10, 6))
plt.hist(errors, bins=30)
plt.title("Distribution of Prediction Errors")
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

#Function for predictive graphes
#This function creates a base beer profile using mean values for all features
def build_base_profile(X_BeerData: pd.DataFrame) -> dict:
    """Builds a mean baseline beer profile."""
    base = {}
    for col in X_BeerData.columns:
        if col == "Style":
            base[col]= int(X_BeerData[col].mode()[0])
        else:
           
            base[col] =float(X_BeerData[col].mean())
    return base


#This function predicts a full grid of values by varying two features at a time
def predict_grid(model_obj, X_BeerData, feature_x, feature_y, x_values, y_values):
    """
    Generates a matrix of predicted ratings by varying 2 features.
    All other features remain fixed at their average values.
    """
    base_profile = build_base_profile(X_BeerData)
    rows = []

    #We generate all combinations of x and y values
    for yv in y_values:
        for xv in x_values:
            sample = base_profile.copy()
            sample[feature_x] =xv
            sample[feature_y] =yv
            rows.append(sample)

    #We predict the rating of each synthetic profile
    grid_BeerData = pd.DataFrame(rows)[X_BeerData.columns]
    preds =model_obj.predict(grid_BeerData)

    #We reshape predictions into a matrix for heatmap visualization
    pred_matrix= preds.reshape(len(y_values), len(x_values))
    return pred_matrix


#This function plots a heatmap showing predicted ratings for 2 selected features
def plot_prediction_heatmap(model_obj, X_BeerData, feature_x, feature_y, title):
    """
    Displays a heatmap with predicted ratings.
    """
    #We use the real min and max values of each feature
    x_min, x_max= X_BeerData[feature_x].min(), X_BeerData[feature_x].max()
    y_min, y_max= X_BeerData[feature_y].min(), X_BeerData[feature_y].max()

    #We create regular intervals to explore the feature space
    x_values =np.linspace(x_min, x_max, 25)
    y_values=np.linspace(y_min, y_max, 25)

    #We compute predictions for each pair of values
    pred_matrix = predict_grid(model_obj, X_BeerData, feature_x, feature_y, x_values, y_values)

    #We display the prediction matrix as a heatmap
    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        pred_matrix,
        origin="lower",
        aspect="auto",
        extent=[x_values.min(), x_values.max(), y_values.min(), y_values.max()]
    )
    plt.colorbar(im, label="Predicted Rating")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(title)
    plt.tight_layout()
    plt.show()


#This function generates synthetic beer profiles and returns the best ones according to the model
def top_predicted_profiles(model_obj, X_BeerData, selected_features, levels_dict, top_n=10):
    """
    Generates synthetic profiles and returns the best ones according to predicted rating.
    """
    import itertools

    #We start from the average beer profile
    base_profile = build_base_profile(X_BeerData)

    #We create all combinations from the selected feature levels
    value_lists=[levels_dict[f] for f in selected_features]
    combinations=list(itertools.product(*value_lists))

    results = []
    for combo in combinations:
        sample = base_profile.copy()

        #We replace selected feature values with the current combination
        for feat, val in zip(selected_features,combo):
            sample[feat]=val

        #We predict the rating of the synthetic beer profile
        sample_BeerData = pd.DataFrame([sample])[X_BeerData.columns]
        pred = model_obj.predict(sample_BeerData)[0]

        #We store the combination and its predicted rating
        row = {feat: val for feat, val in zip(selected_features,combo)}
        row["Predicted Rating"]=pred
        results.append(row)

    #We sort combinations by predicted rating and keep only the best ones
    results_BeerData = pd.DataFrame(results).sort_values(
        by="Predicted Rating",ascending=False
    ).head(top_n)

    return results_BeerData


#This function plots the best predicted synthetic beer profiles
def plot_top_profiles(results_BeerData):
    """
    Displays the best predicted combinations with their predicted ratings.
    """
    labels = []
    values = results_BeerData["Predicted Rating"].values

    for _, row in results_BeerData.iterrows():
        label = (
            f"Fruits={row.get('Fruits', '-')}, "
            f"Sour={row.get('Sour', '-')}, "
            f"Body={row.get('Body', '-')}, "
            f"Malty={row.get('Malty', '-')}"
        )
        labels.append(label)

    plt.figure(figsize=(12, 6))
    bars = plt.barh(labels[::-1], values[::-1])

    for i, v in enumerate(values[::-1]):
        plt.text(v + 0.02, i, f"{v:.3f}", va='center')

    plt.xlabel("Predicted Rating")
    plt.title("Top Predicted Beer Profiles")
    plt.tight_layout()
    plt.show()

#Heatmap
#We analyze how the predicted rating changes depending on Fruits and Sour values
plot_prediction_heatmap(
    model,X,
    feature_x="Fruits",
    feature_y="Sour",
    title="Predicted Rating by Fruits and Sour"
)

#We analyze how the predicted rating changes depending on Body and Malty values
plot_prediction_heatmap(
    model, X,
    feature_x="Body",
    feature_y="Malty",
    title="Predicted Rating by Body and Malty"
)

#We analyze how the predicted rating changes depending on Bitter and Sweet values
plot_prediction_heatmap(
    model,X,
    feature_x="Bitter",
    feature_y="Sweet",
    title="Predicted Rating by Bitter and Sweet"
)

#Top predictive combinaison 
#We generate synthetic beer profiles by varying the most relevant features for this project we decides to use only 4 features beacause of the calculation cost. 
selected_features = ["Fruits", "Sour", "Body", "Malty"]

#We define several realistic levels for each selected feature
levels_dict ={
    "Fruits": list(np.linspace(X["Fruits"].min(), X["Fruits"].max(), 5)),
    "Sour": list(np.linspace(X["Sour"].min(), X["Sour"].max(), 5)),
    "Body": list(np.linspace(X["Body"].min(), X["Body"].max(), 5)),
    "Malty": list(np.linspace(X["Malty"].min(), X["Malty"].max(), 5))
}

#We compute the top synthetic beer profiles according to the model predictions
top_profiles=top_predicted_profiles(
    model_obj=model,
    X_BeerData=X,
    selected_features=selected_features,
    levels_dict=levels_dict,
    top_n=10
)

print("\n Top predicted beer profiles")
print(top_profiles, "\n")

#We display the best predicted beer profiles
plot_top_profiles(top_profiles)
