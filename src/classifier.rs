type DataEntry = (Vec<String>, String);
pub type Data = Vec<DataEntry>;

const REQUIRED_CLEARANCE: f64 = 0.99;
use std::collections::HashMap;

pub enum NodeType {
    Leaf(f64),
    Stage
}

pub struct TreeNode {
    pub node_type: NodeType,
    pub children: Option<HashMap<String, TreeNode>>,
    pub attribute: Option<u8>,
}

pub struct DecisionTreeeClassifier {
    pub root: TreeNode,
}

impl DecisionTreeeClassifier {
    pub fn from_data(data: &Data) -> DecisionTreeeClassifier {
        DecisionTreeeClassifier {
            root: build_tree(data)
        }
    }
    pub fn predict(&self, record: &Vec<String>) -> f64 {
        let mut node = &self.root;
        loop {
            match &node.node_type {
                NodeType::Leaf(class_proba) => return *class_proba,
                NodeType::Stage => {
                    let attr = node.attribute.unwrap();
                    let value = &record[attr as usize];
                    node = node.children.as_ref().unwrap().get(value).unwrap();
                }
            }
        }
    }
}

fn split_by_attribute(data: &Data, attr: u8) -> HashMap<String, Data> {
    let mut result: HashMap<String, Data> = HashMap::new();
    for (record, class) in data {
        let value = record.get(attr as usize).unwrap();
        if result.contains_key(value) {
            result.get_mut(value).unwrap().push((record.clone(), class.clone()));
        } else {
            result.insert(value.clone(), vec![(record.clone(), class.clone())]);
        }
    }
    result
}

fn calculate_enthropy(data: &Data) -> f64 {
    let mut classes = HashMap::new();
    for (_, class) in data {
        let count = classes.entry(class).or_insert(0);
        *count += 1;
    }
    let mut enthropy = 0.0;
    for (_, count) in &classes {
        let p = *count as f64 / data.len() as f64;
        enthropy -= p * p.log2();
    }
    enthropy
}

fn calculate_enthropy_by_attribute(data: &Data, attr: u8) -> f64 {
    let mut enthropy = 0.0;
    let split_data = split_by_attribute(data, attr);
    for (_, group_data) in &split_data {
        let p = group_data.len() as f64 / data.len() as f64;
        enthropy += p * calculate_enthropy(group_data);
    }
    enthropy
}

fn calculate_split_enthropy(data: &Data, attr: u8) -> f64 {
    let split_data = split_by_attribute(data, attr);
    let mut split_enthropy = 0.0;
    for (_, splitted_data) in &split_data {
        let p = splitted_data.len() as f64 / data.len() as f64;
        let p = p * p.log2();
        split_enthropy -= p
    }
    split_enthropy
}

fn calculate_gain(data: &Data, attr: u8) -> f64 {
    let enthropy = calculate_enthropy(data);
    let enthropy_by_attr = calculate_enthropy_by_attribute(data, attr);
    let split_enthropy = calculate_split_enthropy(data, attr);
    if split_enthropy == 0.0 {
        return 0.0;
    }
    (enthropy - enthropy_by_attr)/split_enthropy
}

fn calculate_data_clearance(data: &Data) -> (f64, String) {
    let mut classes = HashMap::new();
    for (_, class) in data {
        let count = classes.entry(class).or_insert(0);
        *count += 1;
    }
    let max_class_count = classes.values().max().unwrap();
    let max_class = classes.iter().find(|(_, count)| *count == max_class_count).unwrap().0;
    (*max_class_count as f64 / data.len() as f64, max_class.to_string())
}

fn build_tree(data: &Data) -> TreeNode {
    let clearance = calculate_data_clearance(data);
    fn get_prob_from_clearance(clearance: (f64, String)) -> f64 {
        if clearance.1 == "e" {
            clearance.0
        } else {
            1.0 - clearance.0
        }
    }
    if clearance.0 >= REQUIRED_CLEARANCE {
        return TreeNode {
            node_type: NodeType::Leaf(get_prob_from_clearance(clearance)),
            children: None,
            attribute: None,
        }
    }
    let mut best_gain = calculate_gain(data, 0);
    let mut best_attr = 0;
    for attr in 0..data.get(0).unwrap().0.len() {
        let gain = calculate_gain(data, attr as u8);
        if gain > best_gain {
            best_gain = gain;
            best_attr = attr;
        }
    }
    if best_gain == 0.0 {
        return TreeNode {
            node_type: NodeType::Leaf(get_prob_from_clearance(clearance)),
            children: None,
            attribute: None,
        }
    }
    TreeNode {
        node_type: NodeType::Stage,
        children: Some(split_by_attribute(data, best_attr as u8).iter().map(|(attr_value, data)| {
            (attr_value.to_string(), build_tree(data))
        }).collect()),
        attribute: Some(best_attr as u8),
    }
}