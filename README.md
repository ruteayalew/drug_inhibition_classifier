# drug_inhibition_classifier
Data exploration, preparation, and a novel classifier model to predict a drug's inhibition for side effects using a dataset of 1000 drug test results.

## Phase II: Exploratory Data Analysis

### Step 1. Identify Data Types
After reading data2d.txt into a dataframe, I defined a function to identify existing datatypes and the interval of columns they exist in. 

  def get_data_type_intervals(df):
    # Dictionary to store data types and their corresponding intervals
    data_type_intervals = {}
    
    # Initialize variables to track the current data type and its start column
    current_data_type = None
    start_column = None

    # Iterate through the columns and their data types
    for col in df.columns:
        data_type = str(df[col].dtype)  # Get the data type as a string

        if data_type != current_data_type:
            # If the data type changes, update the interval for the previous data type
            if current_data_type is not None:
                data_type_intervals[current_data_type] = (start_column, prev_column)

            # Update the current data type and start column for the new data type
            current_data_type = data_type
            start_column = col

        # Keep track of the previous column
        prev_column = col

    # Update the interval for the last data type encountered
    if current_data_type is not None:
        data_type_intervals[current_data_type] = (start_column, prev_column)

    # Convert the dictionary to a list of tuples
    result = [(data_type, interval[0], interval[1]) for data_type, interval in data_type_intervals.items()]
    return result

Reflection: I confirm my assumptions that only the Name and CAS columns contain categorical object data types. However I notice that the Inhibition column is read as integer rather than the binary data types it is contextualized as. The nAtomLac column is the only one of the remaining columns (noted by the assignment details as computationally-derived attributes) that holds integer values rather than floats.

### Step 2. Identify Missing Values
  '''The two defined functions below use polymorphism to report information regarding the presence
of missing data items in the passed dataframe. Both illustrate a measure of quality for the given data. 

The first function, when given only the dataframe, 
reports the total number and percentages of columns containing missing data and missing data count 
out of the entire dataframe. 

The second function take an additional argument of an integer so as to report back the
columns that contain a specified count of missing data or more.

These functions were aided by ChatGPT'''

  def checkMissingVals(*args):
  #the first function is executed when onyl one argument is passed
      if len(args)==1:
          df = args[0]
          missing_columns = df.columns[df.isnull().any()]
          count_col = len(missing_columns)
          count_missingData = 0
          fraction_missing_col = math.trunc((count_col/(len(df.columns))) * 100)
          fraction_missing_data = 0
  
  
          #for each col with missing values, create a list of each index containing a null
          #also updates total count of missing data values
          for col in missing_columns:
              indices_with_missing_values = df[df[col].isnull()].index.tolist()
              count_missingData = count_missingData + len(indices_with_missing_values)

              #optional statement to print each column with missing values and their indices
              #print(f"Column: {col}, Indices: {indices_with_missing_values}")
  
          #calculate fraction of missing values out of total potential data value counts
          total = (len(df.columns))*(len(df))
          fraction_missing_data = math.trunc((count_missingData/total) * 100)
  
          print("Total number of columns with missing data: " , str(count_col))
          print("Fraction of columns with missing data out of all columns: ", str(fraction_missing_col), "%")
          print("Total number of missing data values: ", str(count_missingData))
          print("Fraction of missing data values out of total potential data values: ", str(fraction_missing_data), "%")
          print("\n")

  #the second function is executed when 2 arguments are passed
      if len(args)==2:
          df = args[0]
          missing_count = args[1]
          #add column to list if contains null count >= given integer
          columns_with_missing_count = df.columns[df.isnull().sum() >= missing_count]
                  
          #for each col, 
          for col in columns_with_missing_count:
              count_missingData = len(df[df[col].isnull()].index.tolist())
              print(col,": ",count_missingData)

  #this specific function inquires about the number of null data items of a specific column and how many
  #this serves helpful when we are concerned with particular columns more than others
  def check_col_for_null(df, col_name):
      missing_columns = df.columns[df.isnull().any()]
      missing_count = 0
      if col_name in missing_columns:
          missing_count = df[col_name].isnull().sum()
          print(col_name, "has", missing_count, " missing values")
      else:
          print(col_name, "does not have missing values.")

Reflection: Thankfully those columns have no missing values so now we are just concerned with handling the missing data of the remaining columns. Before doing so, we will determine our method to measure central tendencies. This requires us to first determine if the data is skewed or symmetric, and find the frequency of outliers if any exist.

To make my code as resuable as possible...I also defined a function to return the indices of columns containing numeric values to later slice with iloc whilst calling the pandas skew function.

  def get_numeric_column_indices(df):
    numeric_columns = df.select_dtypes(include='number').columns
    num_columns = len(df.columns)
    iloc_index = []
    
    # Get the indices of numeric columns
    indices_numeric_columns = [df.columns.get_loc(col) for col in numeric_columns]
    iloc_index = [min(indices_numeric_columns),max(indices_numeric_columns) ]
    
    
    return iloc_index

### Step 3. Skew report
  df_1.iloc[:, iloc_index[0]:iloc_index[1]].skew()

After attempting to determine skew through iloc calls, I decide to define a method that reports the skew in an orderly fashion

  def skew_report(df):
    iloc_index = get_numeric_column_indices(df) #calls method to get numeric column indices
    col_skew = 0 #individual skew measurement for each column
    #counters to total each type of skew
    total_skew = 0
    left_skew = 0
    right_skew = 0 
    symmetric = 0
    
    #for each column named in the series returned by the index slice, check skew and count frequency
    for col in df.iloc[:, iloc_index[0]:iloc_index[1]]:
        col_skew = df[col].skew()
        if col_skew < 0:
            left_skew = left_skew + 1
        if col_skew > 0:
            right_skew = right_skew + 1
        if col_skew == 0:
            symmetric = symmetric + 1

    total_skew = left_skew + right_skew
    
    print("Skewness report:")
    print("Number of attributes with left skew: ", left_skew)
    print("Number of attributes with right skew: ", right_skew)
    print("Number of attributes with no skew/ are symmetric: ", symmetric)
    print("\n")
    print("Ratio of total skewed attributes to symmetric attributes = ", total_skew, ":", symmetric)    
    
    if symmetric > total_skew:
        print("Therefor, the data is mostly symmetrical")
    if total_skew > symmetric:
        print("Therefor, the data is mostly skewed")
        
### Output of Skew Report Function:
Skewness report:
Number of attributes with left skew:  67
Number of attributes with right skew:  436
Number of attributes with no skew/ are symmetric:  268


Ratio of total skewed attributes to symmetric attributes =  503 : 268
Therefor, the data is mostly skewed

Reflection: After running the skew_report() function, we see that the attributes of the data2d.txt file are overwhelmingly skewed to the right. Given the abundance of skewed attributes, I assume there are a lot of outliers in the data. I use this knowledge to decide on median as the prefered measurement of central tendency because it is not influenced by extreme values.

## Phase III: Data Preperation

## Step 1. Data Selection
Detect columns that contain no unique values to then later exclude them in the new data frame to be created. The reasoning for this is that the columns with no unique values offer no information to our prediction model. This task will also help reduce the data size to be worked with.

  def find_nonunique_columns(df):
    nonunique_list = []
    for column in df.columns:
        if df[column].nunique() == 1:
            nonunique_list.append(column)
    return nonunique_list

  nonunique_list = find_nonunique_columns(df_1)
  print("Total count of columns with no unique values: ",len(nonunique_list))

Now we know the total count of columns with no unique values is significant. Next step will be to exclude them from our first copy of the original dataframe. 

  #method copies the original dataframe passed into a new dataframe used to drop non-unique columns found previouisly
  def remove_nonunique_columns(df):
      df_unique = df
      nonunique_list = find_nonunique_columns(df)
      for col in nonunique_list:
          if col in df_unique.columns: #minor error handling just in case
              df_unique = df_unique.drop([col],axis=1)
      return df_unique
      
Reflection: After Phase II, I learned there was a significant amount of outliers based off the vastly skewed quality of the data. Because of this, I will be using the median of each attribute to fill their null values. This decision is debatable because calculating the median becomes more costly given larger data sets, and it is important to note that for larger cases, an approximation via median intervals is better practice. 

Below is a defined function to fill missing values within each attribute by their corresponding medians using the standard pandas.median function

  def fill_with_median(df): 
      #create new dataframe to save seperately the soon to be altered data with filled nulls
      df_filled = df
      #first select only columns with numeric data
      numeric_columns = df.select_dtypes(include='number').columns
      
      #then pick out numeric columns that have nulls and fill with median
      for col in numeric_columns:
          if df[col].isnull().any():
              df_filled[col] = df[col].fillna(df[col].median())
              
      return df_filled

Before calling the fill funciton, I check the number of nulls remaining after removing non-unique columns and see that the amount of nulls has been reduced significantly!

  checkMissingVals(df_unique)
#### Output:
Total number of columns with missing data:  299
Fraction of columns with missing data out of all columns:  59 %
Total number of missing data values:  5395
Fraction of missing data values out of total potential data values:  1 %

### Step 2. Feature selection
Excluded categoricals and select features with Best ANOVA F-values

  def feature_selection(df):
    #load libraries
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif

    #get dimensions of df for slicing
    rows = len(df.index)
    columns = len(df.columns)
        
    #Identify features and target
    x = df.iloc[:,1:columns]
    y = df.iloc[:,0].values  
    
    #Select Features with Best ANOVA F-values
    fvalue_selector = SelectKBest(f_classif, k = 100)
    x_kbest = fvalue_selector.fit_transform(x,y)
    
    print("After selecting best 3 features:", x_kbest.shape) 

    filter = fvalue_selector.get_support()
    features = array(x.columns.values.tolist())

    print("Selected best 100:")
    print(features[filter])
   
    return features[filter]

### Step 3. Build Naive Classifier
Created a function to assume the target attribute as zero.

  def create_naive_model(df, naive_target_value, target_attribute):
      df_temp = df
      target_name = str(target_attribute)
      naive_model = pd.DataFrame()
      
      #for every row in dataframe, generate a naive value appended to a list
      #this list is to be used in concatenation later
      naive_val_list = [0] * len(df.index)
  
      #Iterate through column of dataframe to find target column and drop
      #this is for concatenating the prior list of naive values and a 
      #slice of dataframe excluding the original inhibition values
      for col in df_temp.columns:
          if col == target_name:
              df_temp = df_temp.drop(columns=[target_name])
      
  
      naive_col_df = pd.Series(naive_val_list, name = target_name)
      naive_model = pd.concat([df_temp,naive_col_df], axis=1, join='outer')
      return naive_model

    class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor ''' 
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # for leaf node
        self.value = value

      class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        ''' constructor '''
        
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree ''' 
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["info_gain"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''

        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")

        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        # return best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        ''' function to compute information gain '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        ''' function to compute entropy '''
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        ''' function to compute gini index '''
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        ''' function to train the tree '''
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        ''' function to predict new dataset '''
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

### Step 4. Train Naive Classifier

  #Here we are training test-split on the training_data
  test_data = training_data
  X = test_data.iloc[:, :-1].values
  Y = test_data.iloc[:, -1].values.reshape(-1,1)
  from sklearn.model_selection import train_test_split
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

  #Here we are fitting the model on the training_data
  classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
  classifier.fit(X_train,Y_train)
  classifier.print_tree()

### Model Evaluation
The baseline performance of the first model is an accuracy score of 0.721 using the sklearn.metrics accuracy_score function. I prevent overfitting by performing feature selection during Phase II. The Decision Tree Induction code also includes an adjustable stopping criteria and split criteria insdide the 'fitting the model' code block. After the initial test with a max_depth of 3 and minimum_sample_split of 3, I then define a function that combines the fitting code block and accuracy test code blocks to automate different tests with varied inputs for the stopping & splitting criteria. Thus the accuracy with regard to complexity can be compared by using the defined function below with different max_depth and accuracy_score input. 

  def fitModel(training_data,max_depth_val, minimum_sample_split_val):    
    #Here we are training test-split on the training_data
    test_data = training_data
    X = test_data.iloc[:, :-1].values
    Y = test_data.iloc[:, -1].values.reshape(-1,1)
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)
    
    #Here we are fitting the model on the training_data
    classifier = DecisionTreeClassifier(min_samples_split=max_depth_val, max_depth=minimum_sample_split_val)
    classifier.fit(X_train,Y_train)
    print("PRINTING TREE FOR for max_depth = ", max_depth_val, "minimum_samples_split= ", minimum_sample_split_val)
    classifier.print_tree()
        
    #Here we are checking accuracy of decision tree model on the training_data
    Y_pred = classifier.predict(X_test) 
    from sklearn.metrics import accuracy_score
    print("\n\n\n ACCURACY SCORE for max_depth = ", max_depth_val, "minimum_samples_split= ", minimum_sample_split_val)
    print(accuracy_score(Y_test, Y_pred))
    
    return decision_tree_model_pkl
