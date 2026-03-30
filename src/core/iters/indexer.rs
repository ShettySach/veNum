pub(crate) struct Indexer<'a> {
    sizes: &'a [usize],
    indices: Vec<usize>,
    current: usize,
    maximum: usize,
}

impl<'a> Indexer<'a> {
    pub(crate) fn new(sizes: &'a [usize]) -> Self {
        Indexer {
            sizes,
            indices: vec![0; sizes.len()],
            current: 0,
            maximum: sizes.iter().product(),
        }
    }
}

impl Iterator for Indexer<'_> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current == self.maximum {
            return None;
        };

        // Clone current indices to return (still needed, but now it's clear why)
        let result = self.indices.clone();

        // Update indices for next iteration
        for i in (0..self.sizes.len()).rev() {
            self.indices[i] += 1;

            if self.indices[i] >= self.sizes[i] {
                self.indices[i] = 0;
            } else {
                break;
            }
        }

        self.current += 1;
        Some(result)
    }
}
