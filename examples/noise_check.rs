fn main() {
    for n in [5usize, 16, 37, 100003] {
        let v = h3_rs::rng::TorchGen::new(42).randn(n);
        println!("{n} {:.9} {:.9} {:.9} {:.9}", v[0], v[1], v[n / 2], v[n - 1]);
    }
}
