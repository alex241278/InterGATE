// Regenerate Supplementary Data 5 from completed and archived analyses.
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const workspace = path.resolve(here, "../..");
const runtime = process.env.CODEX_WORKSPACE_DEPENDENCIES ??
  path.join(os.homedir(), ".cache/codex-runtimes/codex-primary-runtime/dependencies");
const python = path.join(runtime, "python/bin/python3");
const temp = await fs.mkdtemp(path.join(os.tmpdir(), "intergate-supp5-"));
const target = path.resolve(here, "../manuscript/Supplementary Data/Supplementary_Data_5_Review_analyses.xlsx");
const fileForExport = path.join(temp,"workbook.xlsx");
const fileForPublication = path.join(path.dirname(target),"."+path.basename(temp)+".xlsx.tmp");
const review = "revision_CMPB/analysis/";
const primary = "InterGATE/results_v8/08a_paired_predictive_benchmark/summary/";
const sourceList = [];
const add = (category, name, description, file, specification = undefined, allowEmpty = false) =>
  sourceList.push({category, name, description, source: file, path: path.resolve(workspace, file), specification, allowEmpty});
async function exists(file) {
  try { return (await fs.stat(path.resolve(workspace, file))).isFile(); }
  catch (error) { if (error.code === "ENOENT") return false; throw error; }
}
function runHelper(action, args = []) {
  execFileSync(python, ["-B", path.join(here, "workbook_io.py"), action, temp, ...args], {stdio: "inherit"});
}
async function collect() {
  add("Structure", "Structure sensitivity", "Strict directed cores, score weights, selector settings, source masks and fixed-prior baselines. All seed-specific panels contain 529 directed edges.", review+"r1_structure/structure_summary.csv");
  add("Structure", "Core composition", "Directed candidate identifiers and endpoint genes for each structural specification.", review+"r1_structure/structure_core_edges.csv.gz");
  add("Structure", "Seed selections", "One row per selected candidate, seed and specification. Fixed and aggregate panels appear under fixed and aggregate.", review+"r1_structure/structure_seed_panels.csv.gz");
  add("Structure", "Prior tie sensitivity", "One thousand random resolutions of ties at the cutoff for each deterministic prior-only ranking.", review+"r1_structure/prior_random_tie_audit.csv.gz");
  add("STRING", "STRING coverage", "Coverage of the 214 unordered primary core pairs across STRING thresholds and matched null panels.", review+"r1_structure/string_coverage_vs_matched_prior.csv");
  add("STRING", "STRING core scores", "Highest physical-interaction score for each primary core pair. Zero indicates no mapped physical link in the resource.", review+"r1_structure/string_core_coverage.csv");
  add("STRING", "STRING rankings", "Independent rankings of 529 unordered STRING pairs by confidence, degree or development expression context.", review+"r1_structure/string_independent_rankings.csv");
  add("STRING", "STRING selected pairs", "Unordered pairs selected in the three STRING ranking controls at minimum physical score 700.", review+"r1_structure/string_selected_pairs.csv.gz");
  add("Programmes", "Programme contrasts", "Direct between-panel differences for seven locked programmes. Exact and matched-null probabilities were adjusted across these seven tests separately.", review+"programmes/direct_programme_contrasts.csv");
  add("Cohorts", "Edge concordance", "Development–external correlations and sign agreement by comparator, cohort, adjustment and edge unit.", review+"r2_cohorts/edge_concordance.csv");
  add("Cohorts", "Nested external concordance", "Archived external effects in development-selected nested cores. Counts passing FDR use the original family of 214 unordered pairs in each cohort or meta-analysis.", review+"r2_cohorts/nested_external_concordance.csv");
  if (await exists(review+"r2_cohorts/nested_20_pairs.csv"))
    add("Cohorts", "20-seed external pairs", "All distinct pairs retained across 20 fits, with development and external meta-analytic Fisher-scale contrasts, the original 214-pair adjusted probability and agreement in sign.", review+"r2_cohorts/nested_20_pairs.csv");
  add("Cohorts", "Edge effects", "Cohort-specific and meta-analytic differential-correlation estimates for the 299 fixed directed core edges.", review+"r2_cohorts/edge_effects.csv.gz");
  add("Cohorts", "Contrast denominators", "Numbers of Basal-like and comparator tumours in the development and external cohorts.", review+"r2_cohorts/edge_contrast_denominators.csv");
  add("Cohorts", "TCGA covariate coverage", "Tumours with purity and leukocyte-fraction estimates, including joint coverage, by TCGA subtype.", review+"r2_cohorts/tcga_covariate_coverage.csv");
  add("Harmonisation", "Validation preprocessing", "Change in validation expression after fitting the harmonisation parameters only on structural-training tumours.", review+"r2_cohorts/validation_expression_changes.csv");
  add("Harmonisation", "Validation classifier", "Validation performance of the classifier used to probe the two preprocessing designs. Differences refer to the same patients.", review+"r2_cohorts/validation_classifier.csv");
  add("Harmonisation", "Batch replay", "Maximum absolute expression differences on reproducing the submitted full-batch harmonisation.", review+"r2_cohorts/harmonization_replay.csv");
  add("Harmonisation", "Model replay", "Agreement of replayed seed-level probabilities and predictions with the archived InterGATE and XGBoost results.", review+"r2_cohorts/frozen_model_replay.csv");
  add("Harmonisation", "Composition sensitivity", "Cohort-level effects of changing the reference composition of the target batch around fixed sentinel tumours.", review+"r2_cohorts/external_composition_summary.csv");
  add("Harmonisation", "Sentinel evaluations", "Changes in expression and predicted probabilities per sentinel evaluation. The same tumour may appear more than once.", review+"r2_cohorts/external_composition_sentinel_results.csv");
  for (const [fileModel, title] of [["InterGATE","InterGATE"],["xgboost","XGBoost"]]) {
    add("Harmonisation",title+" sentinels","Cohort summaries of sensitivity to target-batch composition in the frozen "+title+" ensemble.", review+"r2_cohorts/sentinel_summary_"+fileModel+".csv");
    add("Harmonisation",title+" probabilities","Predicted class probabilities from the frozen "+title+" ensemble for each sentinel, replicate and target-batch background.", review+"r2_cohorts/sentinel_probabilities_"+fileModel+".csv.gz");
  }
  add("Prediction", "Benchmark estimates", "Original five-seed ensemble estimates with percentile intervals. Each row identifies its evaluation split, scope and metric.",primary+"bootstrap_overall_95ci.csv");
  add("Prediction", "Benchmark contrasts", "Original InterGATE-minus-comparator differences on identical patients, with paired bootstrap intervals.",primary+"paired_model_differences_95ci.csv");
  add("Prediction", "Architecture estimates", "Review-time five-seed architecture comparisons. Differences and intervals use the same cohort-by-class-stratified patient resamples.",review+"branches/paired_external.csv");
  if (await exists(review+"branches/masking_reliance.csv"))
    add("Prediction","Frozen branch reliance","Mean change after branch neutralisation by density, with uncertainty across five fitted seeds on the fixed validation set.",review+"branches/masking_reliance.csv");
  if (await exists(review+"branches/graph_branch_audit.csv"))
    add("Prediction","Graph branch fits","Fit and checkpoint audit for the separately trained graph branch.",review+"branches/graph_branch_audit.csv");
  add("GeSubNet","GeSubNet comparison","Side-by-side rankings at three panel densities in the matched 113-gene, 397-pair domain.",review+"editor/gesubnet_panel_metrics.csv");
  add("GeSubNet","GeSubNet scores","Per-seed scores, aggregate rankings and archived differential-correlation effects for every shared unordered pair.",review+"editor/gesubnet_pair_scores.csv");
  add("GeSubNet","GeSubNet selections","Selected unordered pairs by method and density in the common domain.",review+"editor/gesubnet_selected_pairs.csv");
  for (const n of [10,20]) {
    const base=review+"r1_structure/extended_"+n;
    const parts=[[" summary","_summary.csv","Nested strict core, structural stability, signature recovery and overlap with the original five-seed core."],
                 [" core","_core_edges.csv.gz","Directed core edges and endpoint genes in the nested seed experiment."],
                 [" selections","_seed_panels.csv.gz","Individual candidate-edge selections in the nested development-only seed experiment."]];
    if ((await Promise.all(parts.map(([,suffix])=>exists(base+suffix)))).every(Boolean))
      for (const [title,suffix,description] of parts)
        add("Extended seeds",n+" seeds"+title,description,base+suffix,"nested_"+n,title===" core");
  }
  const stringRoot=review+"r1_structure/string_training/";
  const items=[["STRING fitted core","five_seed_summary.csv","Five-seed trained-model summary, degree-matched enrichment and directed and unordered overlap against the original core."],
               ["STRING fitted edges","five_seed_core_edges.csv.gz","Strict edge intersection after retraining the five models on the substituted physical prior."],
               ["STRING fitted panels","seed_panels.csv.gz","Directed edge selections for each of the five models trained on STRING."],
               ["STRING fitted overlap","pairwise_overlap.csv","Pairwise directed Jaccard coefficients for the five models trained on STRING."],
               ["STRING development","development_performance.csv","Validation macro-F1 for each fit trained on the alternative prior."]];
  if ((await Promise.all(items.map(([,file])=>exists(stringRoot+file)))).every(Boolean))
    for (const [name,file,description] of items)
      add("STRING training",name,description,stringRoot+file,undefined,name==="STRING fitted edges");
}

function numericStyle(header, values) {
  if (!values.some(v => typeof v==="number") || values.some(v=>typeof v==="string"||typeof v==="boolean")) return undefined;
  if (/^(exact_p|matched_p|exact_BH7|matched_BH7)$/.test(header)) return "0.000E+00";
  if (/recovery|jaccard|rho|spearman|mean|prob|estimate|f1|ci_|ci0|ci9|fold|fraction|effect|delta|agreement|density|score|margin/i.test(header)
      && !/^(STRING_score|string_score|STRING_minimum_score)$/.test(header)) return "0.000000";
  if (values.every(v=>v===null||Number.isInteger(v))) return "#,##0";
  return "0.000000";
}

const palette = {"Structure":"#344A70","STRING":"#3D626C","Programmes":"#6D4E71",
  "Cohorts":"#355B50","Harmonisation":"#486577","Prediction":"#505E7D","GeSubNet":"#665A48",
  "Extended seeds":"#344A70","STRING training":"#3D626C"};
const readJson = async file => JSON.parse(await fs.readFile(file,"utf8"));
const block = (sheet,row,col,height,width) => sheet.getRangeByIndexes(row,col,height,width);
function tabulate(sheet,data,color) {
  const header=data[0], n=data.length, m=header.length;
  sheet.showGridLines=false;
  sheet.tabColor=color;
  for(let i=0;i<n;i+=4000) {
    const chunk=data.slice(i,i+4000);
    block(sheet,i,0,chunk.length,m).values=chunk;
  }
  const top=block(sheet,0,0,1,m);
  top.format={fill:color,font:{name:"Arial",size:10,bold:true,color:"#FFFFFF"},
              wrapText:true,verticalAlignment:"center"};
  top.format.rowHeight=51;
  for(let j=0;j<m;j++) {
    const sample=data.slice(1,Math.min(n,301)).map(row=>row[j]);
    const maxText=Math.max(0,...sample.filter(v=>typeof v==="string").map(s=>s.length));
    const label=header[j], minimum=(j===0?123:101);
    const width=Math.min(maxText>46?440:317,Math.max(minimum,Math.min(label.length,32)*7+24,
                          Math.min(maxText,64)*6.9+18));
    sheet.getCell(0,j).format.columnWidthPx=width;
    const fmt=numericStyle(label,sample);
    if(n>1 && fmt) block(sheet,1,j,n-1,1).setNumberFormat(fmt);
  }
  sheet.freezePanes.freezeRows(1);
}

async function build() {
  execFileSync(python, ["-B",path.join(here,"r2_cohorts","nested_external.py")], {stdio:"inherit"});
  await collect();
  await fs.writeFile(path.join(temp,"sources.json"),JSON.stringify(sourceList));
  runHelper("snapshot");
  await fs.symlink(path.join(runtime,"node/node_modules"),path.join(temp,"node_modules"),"dir");
  await fs.writeFile(path.join(temp,"load.mjs"),'export { Workbook, SpreadsheetFile } from "@oai/artifact-tool";');
  const {Workbook,SpreadsheetFile}=await import(pathToFileURL(path.join(temp,"load.mjs")));
  const workbook=Workbook.create();
  const guide=workbook.worksheets.add("Guide");
  guide.showGridLines=false; guide.tabColor="#243650";
  const guideRows=[
    ["Supplementary Data 5. Review analyses",null,null],
    ["Study","InterGATE subtype-associated breast cancer networks",null],
    ["Reading the tables","Each sheet retains the source field names and numerical estimates. A blank value denotes an undefined or inapplicable estimate.",null],
    ["Analysis units","Primary panels comprise 529 directed edges. The strict five-seed core comprises 299 directed edges or 214 unordered pairs. STRING coverage and GeSubNet use unordered pairs.",null],
    ["Interpretation","Five-seed and matched-null results condition on the stated cohorts, candidate graph and selection procedure. Uncertainty units are given by the calculation reported on each sheet.",null],
    ["Worksheet","Contents and analysis unit","Source table within the InterGATE revision package"],
    ...sourceList.map(s=>[s.name,s.description,s.source+(s.specification?"  ["+s.specification+"]":"")])
  ];
  await fs.writeFile(path.join(temp,"guide.json"),JSON.stringify(guideRows));
  block(guide,0,0,guideRows.length,3).values=guideRows;
  for(let row=0;row<5;row++) {
    if(row===0) guide.mergeCells("A1:C1");
    else guide.mergeCells("B"+(row+1)+":C"+(row+1));
  }
  guide.getCell(0,0).format.columnWidthPx=220;
  guide.getCell(0,1).format.columnWidthPx=590;
  guide.getCell(0,2).format.columnWidthPx=480;
  const all=block(guide,0,0,guideRows.length,3);
  all.format.font={name:"Arial",size:10,color:"#182A3D"};
  all.format.wrapText=true; all.format.verticalAlignment="center";
  all.format.rowHeight=31;
  guide.getRange("A1:C1").format.font={name:"Arial",bold:true,size:16,color:"#243650"};
  guide.getRange("A1:C1").format.rowHeight=35;
  guide.getRange("A6:C6").format={fill:"#243650",font:{name:"Arial",bold:true,size:10,color:"#FFFFFF"},verticalAlignment:"center"};
  guide.getRange("A6:C6").format.rowHeight=25;
  for(let i=0;i<sourceList.length;i++) {
    const row=i+6;
    if(!i||sourceList[i].category!==sourceList[i-1].category)
      block(guide,row,0,1,3).format.borders={top:{style:"thin",color:"#A9BBC6"}};
    guide.getCell(row,0).format.font={name:"Arial",bold:true,size:10,color:palette[sourceList[i].category]};
  }
  for(let i=0;i<sourceList.length;i++) {
    const spec=sourceList[i];
    const grid=await readJson(path.join(temp,i+".json"));
    tabulate(workbook.worksheets.add(spec.name),grid,palette[spec.category]);
  }
  workbook.recalculate();
  await workbook.inspect({kind:"region",sheetId:"Programme contrasts",range:"A1:F8",include:"values,formulas",tableMaxRows:8,tableMaxCols:6,maxChars:700});
  await workbook.inspect({kind:"match",searchTerm:"#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",options:{useRegex:true,maxResults:4},maxChars:350});
  if (process.env.INTERGATE_WORKBOOK_PREVIEW_DIR) {
    const directory=process.env.INTERGATE_WORKBOOK_PREVIEW_DIR;
    await fs.mkdir(directory,{recursive:true});
    for(const [i,sheetName] of ["Guide",...sourceList.map(s=>s.name)].entries()) {
      const shot=await workbook.render({sheetName,range:i===0?"A1:C12":"A1:D6",format:"png",scale:1});
      await fs.writeFile(path.join(directory,String(i).padStart(2,"0")+".png"),new Uint8Array(await shot.arrayBuffer()));
    }
  }
  const binary=await SpreadsheetFile.exportXlsx(workbook);
  await binary.save(fileForExport);
  runHelper("verify",[fileForExport]);
  await fs.copyFile(fileForExport,fileForPublication);
  await fs.rename(fileForPublication,target);
  console.log("Supplementary Data 5 updated.");
}

try { await build(); }
finally {
  await fs.rm(fileForPublication,{force:true});
  await fs.rm(temp,{force:true,recursive:true});
}
