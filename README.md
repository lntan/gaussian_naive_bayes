### Use case

Given that a Iris dataset as the following

| Sepal length        | Sepal width           | Petal length  |Petal width|Species|
| ------------- |:-------------:| -----:|-----:|-----:|
|6.8|3.2|5.9|2.3|virginica|
|5.1|3.5|1.4|0.3|setosa|
|4.4|3.2|1.3|0.2|setosa|
|5.8|4|1.2|0.2|setosa|
|6.3|3.3|6|2.5|virginica|
|5.7|4.4|1.5|0.4|setosa|
|6.2|2.2|4.5|1.5|versicolor|
|7.1|3|5.9|2.1|virginica|
|6.4|2.8|5.6|2.1|virginica|
|6|2.9|4.5|1.5|versicolor|
|5.4|3.9|1.3|0.4|setosa|
|6.1|2.8|4|1.3|versicolor|
|4.4|3|1.3|0.2|setosa|
|5.5|4.2|1.4|0.2|setosa|
|6.1|3|4.6|1.4|versicolor|
|5.9|3|5.1|1.8|virginica|
|4.6|3.2|1.4|0.2|setosa|
|4.7|3.2|1.6|0.2|setosa|
|4.8|3|1.4|0.1|setosa|
|5.6|2.5|3.9|1.1|versicolor|
|5.1|3.4|1.5|0.2|setosa|
|5.1|3.8|1.5|0.3|setosa|
|7.9|3.8|6.4|2|virginica|
|6.3|2.5|5|1.9|virginica|
|6.5|3|5.2|2|virginica|
|5.4|3.9|1.7|0.4|setosa|
|5.7|2.5|5|2|virginica|
|5.5|2.5|4|1.3|versicolor|
|5.8|2.8|5.1|2.4|virginica|
|5.4|3|4.5|1.5|versicolor|
|7.6|3|6.6|2.1|virginica|
|7.7|2.8|6.7|2|virginica|
|5.6|2.8|4.9|2|virginica|
|5.1|3.5|1.4|0.2|setosa|
|6.3|3.3|4.7|1.6|versicolor|
|6.2|2.8|4.8|1.8|virginica|
|4.9|3.6|1.4|0.1|setosa|
|6.3|2.5|4.9|1.5|versicolor|
|4.8|3.1|1.6|0.2|setosa|
|6.1|2.8|4.7|1.2|versicolor|
|5.1|3.3|1.7|0.5|setosa|
|6.9|3.1|5.4|2.1|virginica|
|6.1|3|4.9|1.8|virginica|
|5.1|3.7|1.5|0.4|setosa|
|6.8|2.8|4.8|1.4|versicolor|
|5.4|3.4|1.7|0.2|setosa|
|6.8|3|5.5|2.1|virginica|
|5.6|2.9|3.6|1.3|versicolor|
|5.1|2.5|3|1.1|versicolor|
|6.5|3|5.5|1.8|virginica|
|6.5|2.8|4.6|1.5|versicolor|
|5.8|2.7|5.1|1.9|virginica|
|6.5|3.2|5.1|2|virginica|
|6.9|3.1|4.9|1.5|versicolor|
|5|3|1.6|0.2|setosa|
|5.6|2.7|4.2|1.3|versicolor|
|6.3|2.8|5.1|1.5|virginica|
|5|3.4|1.5|0.2|setosa|
|5.7|3|4.2|1.2|versicolor|
|6.4|2.7|5.3|1.9|virginica|
|4.9|3.1|1.5|0.2|setosa|
|5.2|4.1|1.5|0.1|setosa|
|6.4|3.1|5.5|1.8|virginica|
|6.4|2.8|5.6|2.2|virginica|
|5.7|2.8|4.5|1.3|versicolor|
|6|3.4|4.5|1.6|versicolor|
|5.8|2.7|4.1|1|versicolor|
|6.9|3.2|5.7|2.3|virginica|
|6.4|3.2|5.3|2.3|virginica|
|5|3.5|1.6|0.6|setosa|
|7|3.2|4.7|1.4|versicolor|
|7.2|3|5.8|1.6|virginica|
|5.3|3.7|1.5|0.2|setosa|
|5.6|3|4.1|1.3|versicolor|
|4.8|3|1.4|0.3|setosa|
|6.7|3.3|5.7|2.1|virginica|
|5.2|3.4|1.4|0.2|setosa|
|6.9|3.1|5.1|2.3|virginica|
|5.1|3.8|1.9|0.4|setosa|
|6.2|2.9|4.3|1.3|versicolor|
|6.4|2.9|4.3|1.3|versicolor|
|7.7|3.8|6.7|2.2|virginica|
|6.6|2.9|4.6|1.3|versicolor|
|5|3.5|1.3|0.3|setosa|
|4.4|2.9|1.4|0.2|setosa|
|5.8|2.7|5.1|1.9|virginica|
|5.5|2.6|4.4|1.2|versicolor|
|6.3|3.4|5.6|2.4|virginica|
|6.7|3|5|1.7|versicolor|
|4.7|3.2|1.3|0.2|setosa|
|5|3.3|1.4|0.2|setosa|
|5.8|2.7|3.9|1.2|versicolor|
|4.3|3|1.1|0.1|setosa|
|6.6|3|4.4|1.4|versicolor|
|6.7|2.5|5.8|1.8|virginica|
|5.7|2.6|3.5|1|versicolor|
|4.9|3|1.4|0.2|setosa|
|5.7|3.8|1.7|0.3|setosa|
|4.9|2.5|4.5|1.7|virginica|
|5.1|3.8|1.6|0.2|setosa|

The requirement is predicting the categories of data in the table below

| Sepal length        | Sepal width           | Petal length  |Petal width|Species|
| ------------- |:-------------:| -----:|-----:|-----:|
|6.3|2.3|4.4|1.3|?|
|6|3|4.8|1.8|?|
|5.9|3.2|4.8|1.8|?|
|6.7|3.1|4.7|1.5|?|
|4.6|3.1|1.5|0.2|?|
|6.1|2.6|5.6|1.4|?|
|4.6|3.4|1.4|0.3|?|
|7.7|3|6.1|2.3|?|
|5.7|2.9|4.2|1.3|?|
|5|3.2|1.2|0.2|?|
|5.4|3.4|1.5|0.4|?|
|5|3.6|1.4|0.2|?|
|7.7|2.6|6.9|2.3|?|
|5.2|3.5|1.5|0.2|?|
|5.5|2.3|4|1.3|?|
|5.6|3|4.5|1.5|?|
|6|2.2|4|1|?|
|4.9|3.1|1.5|0.1|?|
|4.8|3.4|1.9|0.2|?|
|6.3|2.7|4.9|1.8|?|
|6.7|3.1|4.4|1.4|?|
|5.2|2.7|3.9|1.4|?|
|5.9|3|4.2|1.5|?|
|5.4|3.7|1.5|0.2|?|
|7.2|3.2|6|1.8|?|
|6|2.2|5|1.5|?|
|6.7|3.1|5.6|2.4|?|
|6.2|3.4|5.4|2.3|?|
|5.5|2.4|3.7|1|?|
|6.3|2.9|5.6|1.8|?|
|6.4|3.2|4.5|1.5|?|
|4.6|3.6|1|0.2|?|
|6.7|3|5.2|2.3|?|
|5.5|3.5|1.3|0.2|?|
|5.8|2.6|4|1.2|?|
|6.5|3|5.8|2.2|?|
|4.8|3.4|1.6|0.2|?|
|6.7|3.3|5.7|2.5|?|
|5.5|2.4|3.8|1.1|?|
|5.7|2.8|4.1|1.3|?|
|5|3.4|1.6|0.4|?|
|6.1|2.9|4.7|1.4|?|
|4.9|2.4|3.3|1|?|
|7.4|2.8|6.1|1.9|?|
|6|2.7|5.1|1.6|?|
|4.5|2.3|1.3|0.3|?|
|5|2|3.5|1|?|
|7.2|3.6|6.1|2.5|?|
|5|2.3|3.3|1|?|
|7.3|2.9|6.3|1.8|?|

### How to resolve this issue using Gaussian Naive Bayes
#### Install gem
`gem install gaussian_naive_bayes`
#### Prepare the training set
```
training_set = [[6.8,3.2,5.9,2.3],[5.1,3.5,1.4,0.3],[4.4,3.2,1.3,0.2],[5.8,4,1.2,0.2],[6.3,3.3,6,2.5],[5.7,4.4,1.5,0.4],[6.2,2.2,4.5,1.5],[7.1,3,5.9,2.1],[6.4,2.8,5.6,2.1],[6,2.9,4.5,1.5],[5.4,3.9,1.3,0.4],[6.1,2.8,4,1.3],[4.4,3,1.3,0.2],[5.5,4.2,1.4,0.2],[6.1,3,4.6,1.4],[5.9,3,5.1,1.8],[4.6,3.2,1.4,0.2],[4.7,3.2,1.6,0.2],[4.8,3,1.4,0.1],[5.6,2.5,3.9,1.1],[5.1,3.4,1.5,0.2],[5.1,3.8,1.5,0.3],[7.9,3.8,6.4,2],[6.3,2.5,5,1.9],[6.5,3,5.2,2],[5.4,3.9,1.7,0.4],[5.7,2.5,5,2],[5.5,2.5,4,1.3],[5.8,2.8,5.1,2.4],[5.4,3,4.5,1.5],[7.6,3,6.6,2.1],[7.7,2.8,6.7,2],[5.6,2.8,4.9,2],[5.1,3.5,1.4,0.2],[6.3,3.3,4.7,1.6],[6.2,2.8,4.8,1.8],[4.9,3.6,1.4,0.1],[6.3,2.5,4.9,1.5],[4.8,3.1,1.6,0.2],[6.1,2.8,4.7,1.2],[5.1,3.3,1.7,0.5],[6.9,3.1,5.4,2.1],[6.1,3,4.9,1.8],[5.1,3.7,1.5,0.4],[6.8,2.8,4.8,1.4],[5.4,3.4,1.7,0.2],[6.8,3,5.5,2.1],[5.6,2.9,3.6,1.3],[5.1,2.5,3,1.1],[6.5,3,5.5,1.8],[6.5,2.8,4.6,1.5],[5.8,2.7,5.1,1.9],[6.5,3.2,5.1,2],[6.9,3.1,4.9,1.5],[5,3,1.6,0.2],[5.6,2.7,4.2,1.3],[6.3,2.8,5.1,1.5],[5,3.4,1.5,0.2],[5.7,3,4.2,1.2],[6.4,2.7,5.3,1.9],[4.9,3.1,1.5,0.2],[5.2,4.1,1.5,0.1],[6.4,3.1,5.5,1.8],[6.4,2.8,5.6,2.2],[5.7,2.8,4.5,1.3],[6,3.4,4.5,1.6],[5.8,2.7,4.1,1],[6.9,3.2,5.7,2.3],[6.4,3.2,5.3,2.3],[5,3.5,1.6,0.6],[7,3.2,4.7,1.4],[7.2,3,5.8,1.6],[5.3,3.7,1.5,0.2],[5.6,3,4.1,1.3],[4.8,3,1.4,0.3],[6.7,3.3,5.7,2.1],[5.2,3.4,1.4,0.2],[6.9,3.1,5.1,2.3],[5.1,3.8,1.9,0.4],[6.2,2.9,4.3,1.3],[6.4,2.9,4.3,1.3],[7.7,3.8,6.7,2.2],[6.6,2.9,4.6,1.3],[5,3.5,1.3,0.3],[4.4,2.9,1.4,0.2],[5.8,2.7,5.1,1.9],[5.5,2.6,4.4,1.2],[6.3,3.4,5.6,2.4],[6.7,3,5,1.7],[4.7,3.2,1.3,0.2],[5,3.3,1.4,0.2],[5.8,2.7,3.9,1.2],[4.3,3,1.1,0.1],[6.6,3,4.4,1.4],[6.7,2.5,5.8,1.8],[5.7,2.6,3.5,1],[4.9,3,1.4,0.2],[5.7,3.8,1.7,0.3],[4.9,2.5,4.5,1.7],[5.1,3.8,1.6,0.2]]
target_category = ["virginica","setosa","setosa","setosa","virginica","setosa","versicolor","virginica","virginica","versicolor","setosa","versicolor","setosa","setosa","versicolor","virginica","setosa","setosa","setosa","versicolor","setosa","setosa","virginica","virginica","virginica","setosa","virginica","versicolor","virginica","versicolor","virginica","virginica","virginica","setosa","versicolor","virginica","setosa","versicolor","setosa","versicolor","setosa","virginica","virginica","setosa","versicolor","setosa","virginica","versicolor","versicolor","virginica","versicolor","virginica","virginica","versicolor","setosa","versicolor","virginica","setosa","versicolor","virginica","setosa","setosa","virginica","virginica","versicolor","versicolor","versicolor","virginica","virginica","setosa","versicolor","virginica","setosa","versicolor","setosa","virginica","setosa","virginica","setosa","versicolor","versicolor","virginica","versicolor","setosa","setosa","virginica","versicolor","virginica","versicolor","setosa","setosa","versicolor","setosa","versicolor","virginica","versicolor","setosa","setosa","virginica","setosa"]

```

#### Train
```
require "gaussian_naive_bayes"

learner = GaussianNaiveBayes::Learner.new
training_set.each_with_index do |vector, index|
  learner.train(vector, target_category[index])
end
```

#### Prepare the test set
```
test_set = [[6.3,2.3,4.4,1.3],[6,3,4.8,1.8],[5.9,3.2,4.8,1.8],[6.7,3.1,4.7,1.5],[4.6,3.1,1.5,0.2],[6.1,2.6,5.6,1.4],[4.6,3.4,1.4,0.3],[7.7,3,6.1,2.3],[5.7,2.9,4.2,1.3],[5,3.2,1.2,0.2],[5.4,3.4,1.5,0.4],[5,3.6,1.4,0.2],[7.7,2.6,6.9,2.3],[5.2,3.5,1.5,0.2],[5.5,2.3,4,1.3],[5.6,3,4.5,1.5],[6,2.2,4,1],[4.9,3.1,1.5,0.1],[4.8,3.4,1.9,0.2],[6.3,2.7,4.9,1.8],[6.7,3.1,4.4,1.4],[5.2,2.7,3.9,1.4],[5.9,3,4.2,1.5],[5.4,3.7,1.5,0.2],[7.2,3.2,6,1.8],[6,2.2,5,1.5],[6.7,3.1,5.6,2.4],[6.2,3.4,5.4,2.3],[5.5,2.4,3.7,1],[6.3,2.9,5.6,1.8],[6.4,3.2,4.5,1.5],[4.6,3.6,1,0.2],[6.7,3,5.2,2.3],[5.5,3.5,1.3,0.2],[5.8,2.6,4,1.2],[6.5,3,5.8,2.2],[4.8,3.4,1.6,0.2],[6.7,3.3,5.7,2.5],[5.5,2.4,3.8,1.1],[5.7,2.8,4.1,1.3],[5,3.4,1.6,0.4],[6.1,2.9,4.7,1.4],[4.9,2.4,3.3,1],[7.4,2.8,6.1,1.9],[6,2.7,5.1,1.6],[4.5,2.3,1.3,0.3],[5,2,3.5,1],[7.2,3.6,6.1,2.5],[5,2.3,3.3,1],[7.3,2.9,6.3,1.8]]

```

#### Classify
```
classifier = learner.classifier
test_set.each do |vector|
  p classifier.classify(vector)
end
```
