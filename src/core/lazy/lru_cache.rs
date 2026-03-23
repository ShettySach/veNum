use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

pub struct LruCache<K, V>
where
    K: Eq + Hash + Clone,
{
    map: HashMap<K, (V, u64)>,
    order: VecDeque<(K, u64)>,
    capacity: usize,
    next_gen: u64,
}

impl<K, V> LruCache<K, V>
where
    K: Eq + Hash + Clone,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            capacity,
            next_gen: 0,
        }
    }

    pub fn get_cloned(&mut self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        if self.map.contains_key(key) {
            self.touch(key);
            self.map.get(key).map(|(value, _)| value.clone())
        } else {
            None
        }
    }

    pub fn insert(&mut self, key: K, value: V) {
        if self.capacity == 0 {
            return;
        }

        if let Some((existing, _)) = self.map.get_mut(&key) {
            *existing = value;
            self.touch(&key);
            return;
        }

        if self.map.len() >= self.capacity {
            self.evict_lru();
        }

        let generation = self.bump_generation();
        self.order.push_back((key.clone(), generation));
        self.map.insert(key, (value, generation));
        self.maybe_compact_order();
    }

    fn touch(&mut self, key: &K) {
        if self.map.contains_key(key) {
            let generation = self.bump_generation();
            if let Some((_, current_generation)) = self.map.get_mut(key) {
                *current_generation = generation;
            }
            self.order.push_back((key.clone(), generation));
            self.maybe_compact_order();
        }
    }

    fn evict_lru(&mut self) {
        while let Some((oldest_key, oldest_generation)) = self.order.pop_front() {
            let should_evict = self
                .map
                .get(&oldest_key)
                .is_some_and(|(_, current_generation)| *current_generation == oldest_generation);
            if should_evict {
                self.map.remove(&oldest_key);
                break;
            }
        }
    }

    fn bump_generation(&mut self) -> u64 {
        let generation = self.next_gen;
        self.next_gen = self.next_gen.wrapping_add(1);
        generation
    }

    fn maybe_compact_order(&mut self) {
        let threshold = self.capacity.saturating_mul(4);
        if self.order.len() <= threshold {
            return;
        }
        self.order.retain(|(key, generation)| {
            self.map
                .get(key)
                .is_some_and(|(_, current_generation)| current_generation == generation)
        });
    }
}
