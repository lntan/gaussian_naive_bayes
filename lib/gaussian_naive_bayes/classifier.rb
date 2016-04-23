class Classifier
  attr_reader :categories_summaries, :categories_probabilities
  def initialize(categories_summaries, categories_probabilities)
    @categories_summaries = categories_summaries
    @categories_probabilities = categories_probabilities
  end

  def classify(vector)
    max_ln_category_probability(vector)[0]
  end

  def max_ln_category_probability(vector)
    all_ln_categories_probabilities(vector).
      to_a.
      sort_by{|ln_category_probability| -ln_category_probability[1]}.
      first
  end

  def all_ln_categories_probabilities(vector)
    @categories_summaries.keys.inject({}) do |map, category|
      map[category] = ln_category_probability(vector, category)
      map
    end
  end

  def ln_category_probability(vector, category)
    sum = 0
    vector.each_with_index do |feature_value, feature|
      sum += ln_normal_distribution(feature_value, @categories_summaries[category][feature][:mean], @categories_summaries[category][feature][:standard_deviation])
    end
    sum + Math.log(@categories_probabilities[category])
  end

  def ln_normal_distribution(x, mean, stdev)
    Math.log(1.0/(stdev*Math.sqrt(2*Math::PI))) - ((x - mean)**2)/(2*(stdev**2))
  end
end
