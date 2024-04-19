pub(crate) mod core;
mod server;

fn main() {
    tracing_subscriber::fmt().init();

    server::run_server();
}

