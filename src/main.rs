use eframe::egui;

#[derive(Debug)]
#[derive(PartialEq)]
enum EsmModel {
    Esm2T6_8M,
    Esm2T12_35M,
    Esm2T33_650M,
}

struct ProtRustApp{
    fasta_path: Option<String>,
    selected_model: EsmModel,
    status: String
}

impl ProtRustApp {
    fn model_name(&self) -> &str {
        match self.selected_model {
            EsmModel::Esm2T6_8M => "ESM2 6.8M",
            EsmModel::Esm2T12_35M => "ESM2 12.35M",
            EsmModel::Esm2T33_650M => "ESM2 33.650M",
        }
    }
}

impl Default for ProtRustApp{
    fn default() -> Self {
        Self { fasta_path: None, selected_model: EsmModel::Esm2T33_650M, status: String::from("Ready") }
    }
}

impl eframe::App for ProtRustApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("left_panel").show(ctx, |ui| {
            ui.label("Model:");
            ui.label(self.model_name());
            egui::ComboBox::from_label("Model")
                .selected_text(self.model_name())
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.selected_model, EsmModel::Esm2T6_8M, "ESM2 6.8M");
                    ui.selectable_value(&mut self.selected_model, EsmModel::Esm2T12_35M, "ESM2 12.35M");
                    ui.selectable_value(&mut self.selected_model, EsmModel::Esm2T33_650M, "ESM2 33.650M");
            });
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label(self.model_name());
        });
    }
}

fn main() -> eframe::Result<()> {
    let prot_rust = ProtRustApp::default();
    println!("{:?}", prot_rust.selected_model);
    eframe::run_native(
        "ProtRust",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Box::new(prot_rust)),
    )
}
