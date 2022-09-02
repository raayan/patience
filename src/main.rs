use rand::SeedableRng;
use std::convert::Infallible;
use std::time::{Duration, Instant};

#[derive(Debug, Default, Copy, Clone)]
struct Parameters<T> {
    pub a: T,
    pub b: T,
    pub c: T,
    pub d: T,
}

struct Optimizer {
    rng: rand::rngs::StdRng,
    pub a: tpe::TpeOptimizer,
    pub b: tpe::TpeOptimizer,
    pub c: tpe::TpeOptimizer,
    pub d: tpe::TpeOptimizer,
}

impl Optimizer {
    fn ask(&mut self) -> Result<Parameters<f64>, Infallible> {
        Ok(Parameters {
            a: self.a.ask(&mut self.rng)?,
            b: self.b.ask(&mut self.rng)?,
            c: self.c.ask(&mut self.rng)?,
            d: self.d.ask(&mut self.rng)?,
        })
    }

    fn tell(&mut self, parameters: Parameters<f64>, value: f64) -> Result<(), tpe::TellError> {
        self.a.tell(parameters.a, value)?;
        self.b.tell(parameters.b, value)?;
        self.c.tell(parameters.c, value)?;
        self.d.tell(parameters.d, value)?;
        Ok(())
    }

    fn new(bounds: Parameters<(f64, f64)>) -> Result<Self, tpe::range::RangeError> {
        Ok(Self {
            rng: rand::rngs::StdRng::from_seed(Default::default()),
            a: tpe::TpeOptimizer::new(tpe::parzen_estimator(), tpe::range(bounds.a.0, bounds.a.1)?),
            b: tpe::TpeOptimizer::new(tpe::parzen_estimator(), tpe::range(bounds.b.0, bounds.b.1)?),
            c: tpe::TpeOptimizer::new(tpe::parzen_estimator(), tpe::range(bounds.c.0, bounds.c.1)?),
            d: tpe::TpeOptimizer::new(tpe::parzen_estimator(), tpe::range(bounds.d.0, bounds.d.1)?),
        })
    }
}

fn objective(parameters: Parameters<f64>) -> f64 {
    std::thread::sleep(Duration::from_millis(50));
    let result = parameters.c * (parameters.a / parameters.b) + parameters.d;
    result
}

#[tokio::main]
async fn main() {
    let mut optimizer = Optimizer::new(Parameters {
        a: (0., 1.),
        b: (0., 1.),
        c: (0., 1.),
        d: (0., 1.),
    })
    .unwrap();

    let mut trials_complete = 0;
    let trials = 1000;
    let batch_size = 50;
    let mut best_params = Parameters::<f64>::default();
    let mut best_value = f64::INFINITY;

    println!("running...");
    let start_time = Instant::now();

    while trials_complete < trials {
        // This gives you concurrency
        let batch_fut = (0..batch_size)
            .map(|_| optimizer.ask().unwrap())
            .map(|params| async move {
                match -objective(params) {
                    value if value.is_finite() => (params, value),
                    _ => (params, f64::INFINITY),
                }
            })
            // This gives you actual parallelism
            .map(tokio::spawn);

        let batch = futures::future::join_all(batch_fut)
            .await
            .into_iter()
            .map(|r| r.unwrap());

        for (params, value) in batch {
            optimizer.tell(params, value).unwrap();
            if value < best_value {
                best_params = params;
                best_value = value;
            }
            trials_complete += 1;
        }
    }

    let elapsed_ms = (Instant::now() - start_time).as_millis();

    println!(
        "\
trials      = {trials_complete}
best_params = {best_params:?}
best_value  = {best_value}
elapsed_ms  = {elapsed_ms}\
    "
    );
}
