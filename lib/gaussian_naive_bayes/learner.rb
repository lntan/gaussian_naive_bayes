module GaussianNaiveBayes
  class Learner
    def train(vector, category)
      @category_to_feature_group ||= {}
      @category_to_feature_group[category] ||= {}
      vector.each_with_index do |feature_value, feature|
        @category_to_feature_group[category][feature] ||= []
        @category_to_feature_group[category][feature] << feature_value
      end
      @category_to_num_instances ||= Hash.new(0)
      @category_to_num_instances[category] += 1
    end

    def classifier
      Classifier.new(categories_summaries, categories_probabilities)
    end

    def categories_summaries
      @category_to_feature_group.inject({}) do |map, (category, feature_group)|
        map[category] = category_summary(feature_group)
        map
      end
    end

    def category_summary(feature_group)
      feature_group.inject({}) do |map, (feature, feature_values)|
        map[feature] = {}
        map[feature][:mean] = average(feature_values)
        map[feature][:standard_deviation] = standard_deviation(feature_values)
        map
      end
    end

    def average(numbers)
      numbers.reduce(&:+).to_f/numbers.length
    end

    def standard_deviation(numbers)
      mean = average(numbers)
      variance = numbers.inject(0) do |sum, number|
        sum += (number - mean)**2
      end.to_f/(numbers.length - 1)
      Math.sqrt(variance)
    end

    def categories_probabilities
      total_instances = @category_to_num_instances.values.reduce(&:+)
      @category_to_num_instances.inject({}) do |map, (category, num_instances)|
        map[category] = num_instances.to_f/total_instances
        map
      end
    end
  end
end
