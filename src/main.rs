use rand::Rng;
pub mod classifier;
use classifier::DecisionTreeeClassifier;
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
    let mut correct = 0;
    let mut false_positive = 0;
    let mut false_negative = 0;
    for (record, class) in test_data.iter() {
        let predicted_class = classifier.predict(record);
        if predicted_class == *class {
            correct += 1;
        } else {
            if predicted_class == "e" {
                false_positive += 1;
            } else {
                false_negative += 1;
            }
        }
    }
    println!("Accuracy: {}", correct as f64 / test_data.len() as f64);
    println!("Precision: {}", correct as f64 / (correct + false_positive) as f64);
    println!("Recall: {}", correct as f64 / (correct + false_negative) as f64);
}