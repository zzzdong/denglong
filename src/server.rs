use std::sync::Arc;

use lieweb::{App, AppState, Json, LieRequest, LieResponse, PathParam};

use crate::core::Core;


struct Server {
    
}


#[derive(Debug, Default, serde::Deserialize)]
struct GenerateParameters {
    pub max_new_tokens: usize,
    pub return_full_text: bool,
    pub temperature: f32,
    pub top_p: f32,
}

#[derive(Debug, Default, serde::Deserialize)]
#[serde(default)]
struct GenerateRequest {
    pub inputs: String,
    pub prompt: String,
    pub parameters: GenerateParameters,
}


#[derive(Debug, serde::Serialize)]
struct GenerateResponse {
  pub generated_text: String,
}

impl GenerateResponse {
    pub fn new(generated_text: String) -> Self {
        GenerateResponse { generated_text }
    }
}


#[derive(Debug, Default, serde::Deserialize)]
#[serde(default)]
struct ModelParam {
    pub model: String,
}

async fn hf_generate(state: AppState<Arc<Core>>, param: PathParam<ModelParam>, req: Json<GenerateRequest>) -> LieResponse {
    println!("==> model({}) req({:?})", param.value().model, req.value());
    let mut generation = state.as_ref().new_generation().expect("new genration failed");

    let ret = generation.run(&req.value().inputs, req.value().parameters.max_new_tokens).expect("generation.run() failed");


    LieResponse::with_json(GenerateResponse::new(ret))
}


pub fn run_server() {
    let core = Core::new().expect("create core failed");
    let mut app = App::with_state(Arc::new(core));

    app.post("/api/generate/models/*model", hf_generate);

    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("build runtime failed");

    let ret = rt.block_on(async move {
        println!("ready to run server");
        app.run("localhost:4321").await
    });
    
    if let Err(err) = ret {
        eprintln!("run server error: {err}")
    }
}
