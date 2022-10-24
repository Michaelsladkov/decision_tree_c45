use rand::Rng;
pub mod classifier;
pub mod plotter;
use classifier::DecisionTreeeClassifier;
use classifier::Data;
//Get 5 num different numbers vector, each of them no more than top_border and more than low_border
fn get_random_attributes(num: u8, low_border: u8, top_border: u8) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    let mut attributes = Vec::new();
    for _ in 0..num {
        let mut attribute = rng.gen_range(low_border..top_border);
        while attributes.contains(&attribute) {
            attribute = rng.gen_range(low_border..top_border);
        }
        attributes.push(attribute);
    }
    attributes
}

fn split_data(ratio: f64, data: &Vec<(Vec<String>, String)>) -> (Vec<(Vec<String>, String)>, Vec<(Vec<String>, String)>) {
    let mut rng = rand::thread_rng();
    let mut train_data = Vec::new();
    let mut test_data = Vec::new();
    for (record, class) in data {
        let p = rng.gen_range(0.0..1.0);
        if p < ratio {
            train_data.push((record.clone(), class.clone()));
        } else {
            test_data.push((record.clone(), class.clone()));
        }
    }
    (train_data, test_data)
}

const DATA_PATH: &str = "data\\data.csv";
const NUMBER_OF_ATTRIBUTES: u8 = 5;
const TOTAL_ATTRIBUTES: u8 = 23;
struct ClassifierResults {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub correct: u32,
    pub false_positive_rate: f64,
    pub false_negative_rate: f64,
    pub true_positive_rate: f64,
    pub true_negative_rate: f64,
}
fn bench_classifier(classifier: &DecisionTreeeClassifier, test_data: &Data, threshold: f64) -> ClassifierResults {
    let mut true_positive = 0;
    let mut true_negative = 0;
    let mut false_positive = 0;
    let mut false_negative = 0;
    for (record, class) in test_data.iter() {
        let predicted_proba = classifier.predict(record);
        if predicted_proba >= threshold {
            if class == "e" {
                true_positive += 1;
            } else {
                false_positive += 1;
            }
        } else {
            if class == "p" {
                true_negative += 1;
            } else {
                false_negative += 1;
            }
        }
    }
    let correct = true_positive + true_negative;
    ClassifierResults {
        accuracy: correct as f64 / test_data.len() as f64,
        precision: true_positive as f64 / (true_positive + false_positive) as f64,
        recall: true_positive as f64 / (true_positive + false_negative) as f64,
        correct: correct,
        false_positive_rate: false_positive as f64 / (false_positive + true_negative) as f64,
        false_negative_rate: false_negative as f64 / (false_negative + true_positive) as f64,
        true_positive_rate: true_positive as f64 / (true_positive + false_negative) as f64,
        true_negative_rate: true_negative as f64 / (true_negative + false_positive) as f64,
    }
}
fn main() {
    let mut rdr = csv::ReaderBuilder::new().has_headers(false).from_path(DATA_PATH).unwrap();
    let attrs = get_random_attributes(NUMBER_OF_ATTRIBUTES, 1, TOTAL_ATTRIBUTES);
    println!("{:?}", attrs);
    let mut data = Vec::new();
    for result in rdr.records() {
        let record = result.unwrap();
        let mut record_attrs = Vec::new();
        let class = String::from(record.get(0).unwrap());
        for attr in &attrs {
            record_attrs.push(String::from(record.get(*attr as usize).unwrap()));
        }
        data.push((record_attrs, class));
    }
    let (train_data, test_data) = split_data(0.7, &data);
    let classifier = DecisionTreeeClassifier::from_data(&train_data);
    let results = bench_classifier(&classifier, &test_data, 0.5);
    println!("For threshold 0.5:");
    println!("Accuracy: {}", results.accuracy);
    println!("Precision: {}", results.precision);
    println!("Recall: {}", results.recall);
    println!("False positive rate: {}", results.false_positive_rate);
    println!("False negative rate: {}", results.false_negative_rate);
    println!("True positive rate: {}", results.true_positive_rate);
    println!("True negative rate: {}", results.true_negative_rate);
    let mut roc_series = Vec::new();
    let mut pr_series = Vec::new();
    for threshold in 0..100 {
        let threshold = threshold as f64 / 100.0;
        let results = bench_classifier(&classifier, &test_data, threshold);
        roc_series.push((results.false_positive_rate, results.true_positive_rate));
        pr_series.push((results.precision, results.recall));
    }
    plotter::draw_series(roc_series, "roc.png", "ROC curve");
    plotter::draw_series(pr_series, "pr.png", "PR curve");
}