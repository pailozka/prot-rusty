use eframe::egui;


struct ProtRustApp{
    fasta_path: Option<String>,
    selected_model: String,
    status: String
}

impl Default for ProtRustApp{
    fn default() -> Self {
        Self { fasta_path: None, selected_model: String::from("esm2_t6_8M_UR50D"), status: String::from("Ready") }
    }
}

impl eframe::App for ProtRustApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("ProtRust");
        });
    }
}

fn main() -> eframe::Result<()> {
    let prot_rust = ProtRustApp::default();
    println!("{}", prot_rust.selected_model);
    eframe::run_native(
        "ProtRust",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Box::new(prot_rust)),
    )
}
