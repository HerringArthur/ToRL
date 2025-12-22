import argparse
import torch
import json
import os
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer

# 复用我们已经写好的模块
from model import TrainTicketPolicyModel
from dataset import TrainTicketRLDataset
from mock_env import TrainTicketMockEnv

class EvalConfig:
    """评估专用配置类"""
    def __init__(self, args):
        self.model_path = args.base_model_path  # 基础模型
        self.lora_path = args.checkpoint_path   # 训练好的 LoRA 权重
        self.data_path = args.data_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 评估参数
        self.max_steps_per_episode = 20
        self.temperature = 0.0  # 评估时通常使用 Greedy Search (temp=0) 以求稳定
        self.max_new_tokens = 256
        self.top_k = 1 # Greedy decoding
        
        # 必须与训练时一致的工具定义
        self.TOOLS_LIST = [
            "arxiv-latex-get_paper_prompt",
            "arxiv_-download_paper",
            "arxiv_local-download-paper",
            "arxiv_local-download_paper",
            "arxiv_local-list_papers",
            "arxiv_local-read_paper",
            "arxiv_local-search_papers",
            "bigquery_get_dataset_info",
            "browser_fill",
            "browser_wait_for",
            "canvas-canvas_canvas_health_check",
            "canvas-canvas_create_announcement",
            "canvas-canvas_create_assignment",
            "canvas-canvas_create_conversation",
            "canvas-canvas_create_course",
            "canvas-canvas_create_quiz",
            "canvas-canvas_create_quiz_question",
            "canvas-canvas_create_user",
            "canvas-canvas_enroll_user",
            "canvas-canvas_get_account",
            "canvas-canvas_get_assignment",
            "canvas-canvas_get_course",
            "canvas-canvas_get_course_grades",
            "canvas-canvas_get_dashboard",
            "canvas-canvas_get_dashboard_cards",
            "canvas-canvas_get_quiz",
            "canvas-canvas_get_submission",
            "canvas-canvas_get_syllabus",
            "canvas-canvas_get_upcoming_assignments",
            "canvas-canvas_get_user_grades",
            "canvas-canvas_get_user_profile",
            "canvas-canvas_health_check",
            "canvas-canvas_list_account_courses",
            "canvas-canvas_list_account_users",
            "canvas-canvas_list_announcements",
            "canvas-canvas_list_assignments",
            "canvas-canvas_list_calendar_events",
            "canvas-canvas_list_conversations",
            "canvas-canvas_list_courses",
            "canvas-canvas_list_enrollments",
            "canvas-canvas_list_files",
            "canvas-canvas_list_folders",
            "canvas-canvas_list_quiz_questions",
            "canvas-canvas_list_quizzes",
            "canvas-canvas_list_sub_accounts",
            "canvas-canvas_list_users",
            "canvas-canvas_list_users_in_course",
            "canvas-canvas_start_quiz_attempt",
            "canvas-canvas_submit_assignment",
            "canvas-canvas_submit_assignment_with_file",
            "canvas-canvas_submit_grade",
            "canvas-canvas_submit_quiz_answers",
            "canvas-canvas_update_assignment",
            "canvas-canvas_update_course",
            "canvas-canvas_update_quiz",
            "canvas-canvas_update_quiz_question",
            "canvas-canvas_upload_file_from_path",
            "canvas-ccanvas_list_account_users",
            "canvas-ccanvas_list_announcements",
            "canvas-get_user_profile",
            "canvas-submit_grade",
            "canvas_create_conversation",
            "canvas_get_dashboard_cards",
            "canvas_get_user_profile",
            "canvas_health_check",
            "canvas_list_account_users",
            "canvas_list_courses",
            "check_context_status",
            "dataset_search",
            "emails-check_connection",
            "emails-create_folder",
            "emails-download_attachment",
            "emails-get_email_headers",
            "emails-get_emails",
            "emails-get_folders",
            "emails-read_email",
            "emails-reply_email",
            "emails-save_draft",
            "emails-search_emails",
            "emails-send_-email",
            "emails-send_email",
            "excel-apply_formula",
            "excel-copy_range",
            "excel-copy_worksheet",
            "excel-create_chart",
            "excel-create_pivot_table",
            "excel-create_table",
            "excel-create_workbook",
            "excel-create_works_heet",
            "excel-create_worksheet",
            "excel-delete_range",
            "excel-delete_worksheet",
            "excel-edit_file",
            "excel-format_range",
            "excel-get_workbook_metadata",
            "excel-read_data_from_excel",
            "excel-rename_worksheet",
            "excel-write_data_to_excel",
            "fetch-fetch_html",
            "fetch-fetch_json",
            "fetch-fetch_markdown",
            "fetch-fetch_txt",
            "fetch_html",
            "fetch_json",
            "fetch_markdown",
            "fetch_txt",
            "filesystem--create_directory",
            "filesystem-copy_file",
            "filesystem-create_directory",
            "filesystem-delete_file",
            "filesystem-directory_tree",
            "filesystem-edit_file",
            "filesystem-evaluate",
            "filesystem-get_file_info",
            "filesystem-list_allowed_directories",
            "filesystem-list_directory",
            "filesystem-list_directory_with_sizes",
            "filesystem-move_file",
            "filesystem-read_file",
            "filesystem-read_multiple_files",
            "filesystem-readfila",
            "filesystem-search_files",
            "filesystem-write_file",
            "get_current_date",
            "get_emails",
            "get_file_contents",
            "get_historical_stock_prices",
            "get_sheet_data",
            "get_stock_price_by_date",
            "get_workbook_metadata",
            "git-git_add",
            "git-git_checkout",
            "git-git_commit",
            "git-git_log",
            "git-git_show",
            "git-git_status",
            "git-log",
            "github-add_comment_to_pending_review",
            "github-add_issue_comment",
            "github-create_branch",
            "github-create_gist",
            "github-create_issue",
            "github-create_or_update_file",
            "github-create_pull_request",
            "github-create_repository",
            "github-delete_file",
            "github-fork_repository",
            "github-get_commit",
            "github-get_file_contents",
            "github-get_issue",
            "github-get_issue_comments",
            "github-get_latest_release",
            "github-get_me",
            "github-get_pull_request",
            "github-get_pull_request_diff",
            "github-get_pull_request_files",
            "github-list_branches",
            "github-list_commits",
            "github-list_files",
            "github-list_issues",
            "github-list_notifications",
            "github-list_pull_requests",
            "github-list_releases",
            "github-list_repos",
            "github-list_repositories",
            "github-merge_pull_request",
            "github-push_files",
            "github-rename_repository",
            "github-search_code",
            "github-search_issues",
            "github-search_repositories",
            "github-update_file",
            "github-update_issue",
            "google-cloud-bigquery_create_dataset",
            "google-cloud-bigquery_export_table",
            "google-cloud-bigquery_get_dataset_info",
            "google-cloud-bigquery_list_datasets",
            "google-cloud-bigquery_load_csv_data",
            "google-cloud-bigquery_run_query",
            "google-cloud-logging_create_log_bucket",
            "google-cloud-logging_create_log_sink",
            "google-cloud-logging_export_logs_to_bigquery",
            "google-cloud-logging_list_log_buckets",
            "google-cloud-logging_list_log_sinks",
            "google-cloud-logging_list_logs",
            "google-cloud-logging_read_logs",
            "google-cloud-logging_write_log",
            "google-cloud-storage_copy_object",
            "google-cloud-storage_create_bucket",
            "google-cloud-storage_delete_object",
            "google-cloud-storage_download_file",
            "google-cloud-storage_generate_signed_url",
            "google-cloud-storage_get_bucket_info",
            "google-cloud-storage_list_buckets",
            "google-cloud-storage_list_objects",
            "google-cloud-storage_search_objects",
            "google-cloud-storage_upload_file",
            "google-sheet-update_cells",
            "google_-map-maps_search_places",
            "google_-maps_distance_matrix",
            "google_-maps_search_places",
            "google_calendar-create_event",
            "google_calendar-delete_event",
            "google_calendar-list_events",
            "google_cloud_logging_list_log_buckets",
            "google_forms-add_multiple_choice_question",
            "google_forms-add_text_question",
            "google_forms-create_form",
            "google_forms-get_form",
            "google_forms-get_form_responses",
            "google_map-maps_directions",
            "google_map-maps_distance_matrix",
            "google_map-maps_geocode",
            "google_map-maps_place_details",
            "google_map-maps_reverse_geocode",
            "google_map-maps_search_places",
            "google_sheet-add_rows",
            "google_sheet-batch_update_cells",
            "google_sheet-copy_sheet",
            "google_sheet-create_sheet",
            "google_sheet-create_spreadsheet",
            "google_sheet-format_range",
            "google_sheet-get_multiple_sheet_data",
            "google_sheet-get_multiple_spreadsheet_summary",
            "google_sheet-get_sheet_data",
            "google_sheet-get_sheet_formulas",
            "google_sheet-list_sheets",
            "google_sheet-list_spreadsheets",
            "google_sheet-rename_sheet",
            "google_sheet-share_spreadsheet",
            "google_sheet-update_cells",
            "howtocook-m-c-p_howtocook_get-recipe-by-id",
            "howtocook-m-c-p_howtocook_getRecipeById",
            "howtocook-mcp_howtocook_getAllRecipes",
            "howtocook-mcp_howtocook_getRecipeById",
            "howtocook-mcp_howtocook_getRecipesByCategory",
            "howtocook-mcp_howtocook_recommendMeals",
            "howtocook-mcp_howtocook_whatToEat",
            "howtocoook-mcp_howtocook_getRecipeById",
            "html",
            "huggingface-dataset_search",
            "huggingface-hf_doc_fetch",
            "huggingface-hf_doc_search",
            "huggingface-hf_whoami",
            "huggingface-hub_repo_details",
            "huggingface-model_search",
            "huggingface-paper_search",
            "k-kubectl_create",
            "k-kubectl_describe",
            "k-rollout_restart",
            "k8s-exec_in_pod",
            "k8s-install_helm_chart",
            "k8s-kubectl_apply",
            "k8s-kubectl_context",
            "k8s-kubectl_create",
            "k8s-kubectl_delete",
            "k8s-kubectl_describe",
            "k8s-kubectl_exec",
            "k8s-kubectl_exec_in_pod",
            "k8s-kubectl_generic",
            "k8s-kubectl_get",
            "k8s-kubectl_logs",
            "k8s-kubectl_patch",
            "k8s-kubectl_rollout",
            "k8s-kubectl_scale",
            "k8s-list_api_resources",
            "k8s-ping",
            "k8s-port_forward",
            "k8s-stop_port_forward",
            "k8s-uninstall_helm_chart",
            "k8s-upgrade_helm_chart",
            "kubectl_get",
            "list_allowed_directories",
            "list_directory",
            "list_schemas",
            "list_spreadsheets",
            "list_tables",
            "list_tools",
            "local-browse_history",
            "local-check_context_status",
            "local-claim_done",
            "local-exec",
            "local-manage_context",
            "local-python-execute",
            "local-python_execute",
            "local-read_file",
            "local-search_history",
            "local-search_in_turn",
            "local-search_overlong_tooloutput",
            "local-search_overlong_tooloutput_navigate",
            "local-sleep",
            "local-view_history_turn",
            "local-view_overlong_tooloutput",
            "local-view_overlong_tooloutput_navigate",
            "local-web-search",
            "local-web_search",
            "local-web_search",
            "maps_directions",
            "maps_geocode",
            "memory-add_observations",
            "memory-create_entities",
            "memory-create_relations",
            "memory-delete_entities",
            "memory-delete_observations",
            "memory-open_nodes",
            "memory-read_graph",
            "memory-search_nodes",
            "notion-API-create-a-comment",
            "notion-API-create-a-database",
            "notion-API-delete-a-block",
            "notion-API-get-block-children",
            "notion-API-get-self",
            "notion-API-patch-block-children",
            "notion-API-patch-page",
            "notion-API-post-database-query",
            "notion-API-post-page",
            "notion-API-post-search",
            "notion-API-post_search",
            "notion-API-retrieve-a-block",
            "notion-API-retrieve-a-database",
            "notion-API-retrieve-a-page",
            "notion-API-update-a-block",
            "notion-API-update-a-database",
            "open_nodes",
            "pdf--tools-read_pdf_pages",
            "pdf-tools-extract_pdf_pages",
            "pdf-tools-get_pdf_info",
            "pdf-tools-merge_pdfs",
            "pdf-tools-read_multiple_files",
            "pdf-tools-read_pdf pages",
            "pdf-tools-read_pdf_pages",
            "pdf-tools-search_next_page",
            "pdf-tools-search_pdf_content",
            "pdf-tools-search_pdf_go_page",
            "pdf-tools-search_pdf_next_page",
            "pdf-tools-search_pdf_prev_page",
            "pdf-tools_read_pdf_pages",
            "play-on-active-tab-browser_snapshot",
            "play-on-youtube-with-ocr-get_video_description",
            "playlists_getPlaylistItems",
            "playwright_-browser_snapshot_navigate_to_span",
            "playwright_with-chunk-browser_click",
            "playwright_with-chunk_browser_snapshot_navigate_to_next_span",
            "playwright_with_chunk-browser_click",
            "playwright_with_chunk-browser_close",
            "playwright_with_chunk-browser_console_messages",
            "playwright_with_chunk-browser_evaluate",
            "playwright_with_chunk-browser_handle_dialog",
            "playwright_with_chunk-browser_hover",
            "playwright_with_chunk-browser_install",
            "playwright_with_chunk-browser_key_press",
            "playwright_with_chunk-browser_navigate",
            "playwright_with_chunk-browser_navigate_back",
            "playwright_with_chunk-browser_navigate_forward",
            "playwright_with_chunk-browser_navigate_to_span",
            "playwright_with_chunk-browser_network_requests",
            "playwright_with_chunk-browser_press_key",
            "playwright_with_chunk-browser_resize",
            "playwright_with_chunk-browser_select_option",
            "playwright_with_chunk-browser_snapshot",
            "playwright_with_chunk-browser_snapshot_navigate_to_first_span",
            "playwright_with_chunk-browser_snapshot_navigate_to_last_span",
            "playwright_with_chunk-browser_snapshot_navigate_to_line",
            "playwright_with_chunk-browser_snapshot_navigate_to_next_span",
            "playwright_with_chunk-browser_snapshot_navigate_to_prev_span",
            "playwright_with_chunk-browser_snapshot_navigate_to_span",
            "playwright_with_chunk-browser_snapshot_search",
            "playwright_with_chunk-browser_tab_list",
            "playwright_with_chunk-browser_tab_new",
            "playwright_with_chunk-browser_tab_select",
            "playwright_with_chunk-browser_take_screenshot",
            "playwright_with_chunk-browser_type",
            "playwright_with_chunk-browser_wait_for",
            "playwright_with_chunk_browser_click",
            "playwright_with_chunk_browser_navigate",
            "playwright_with_chunk_browser_snapshot_navigate_to_next_span",
            "playwright_with_chunk_browser_snapshot_navigate_to_span",
            "playwright_with_chunk_browser_snapshot_search",
            "playwright_with_chunk_browser_type",
            "playwright_with_chunk_browser_wait_for",
            "pptx-extract_presentation_text",
            "pptx-get_presentation_info",
            "pptx-list_slide_templates",
            "pptx-open_presentation",
            "rail_1-get-station-code-by-names",
            "rail_12-get-tickets",
            "rail_12306-get-current-date",
            "rail_12306-get-station-by-telecode",
            "rail_12306-get-station-code-by-names",
            "rail_12306-get-station-code-of-citys",
            "rail_12306-get-stations-code-in-city",
            "rail_12306-get-tickets",
            "rail_12306-search-stations-code-in-city",
            "read_data_from_excel",
            "read_file",
            "read_graph",
            "read_pdf_pages",
            "run_code",
            "run_command",
            "scholarly-search-arxiv",
            "scholarly-search-google-scholar",
            "search_emails",
            "search_in_turn",
            "search_nodes",
            "show_security_rules",
            "snowflake-append_insight",
            "snowflake-check_connection",
            "snowflake-create_databases",
            "snowflake-create_schemas",
            "snowflake-create_table",
            "snowflake-describe_table",
            "snowflake-list_databases",
            "snowflake-list_schemas",
            "snowflake-list_tables",
            "snowflake-read_query",
            "snowflake-write-query",
            "snowflake-write_query",
            "terminal-run-command",
            "terminal-run_command",
            "terminal-show_security_rules",
            "view_overlong_tooloutput",
            "wandb-count_weave_traces_tool",
            "wandb-query_wandb_entity_projects",
            "wandb-query_wandb_support_bot",
            "wandb-query_wandb_tool",
            "wandb-query_weave_traces_tool",
            "web_search",
            "woo_customers_list",
            "woo_products_list",
            "woocommerce-woo_coupons_create",
            "woocommerce-woo_coupons_list",
            "woocommerce-woo_customers_create",
            "woocommerce-woo_customers_get",
            "woocommerce-woo_customers_list",
            "woocommerce-woo_customers_update",
            "woocommerce-woo_orders_batch_update",
            "woocommerce-woo_orders_list",
            "woocommerce-woo_orders_notes_create",
            "woocommerce-woo_orders_update",
            "woocommerce-woo_products_batch_update",
            "woocommerce-woo_products_categories_create",
            "woocommerce-woo_products_categories_list",
            "woocommerce-woo_products_create",
            "woocommerce-woo_products_get",
            "woocommerce-woo_products_list",
            "woocommerce-woo_products_tags_list",
            "woocommerce-woo_products_update",
            "woocommerce-woo_products_variations_get",
            "woocommerce-woo_products_variations_list",
            "woocommerce-woo_reports_customers",
            "woocommerce-woo_reports_low_stock",
            "woocommerce-woo_reports_orders",
            "woocommerce-woo_reports_products",
            "woocommerce-woo_reports_sales",
            "woocommerce-woo_reports_top_sellers",
            "woocommerce-woo_settings_get",
            "woocommerce-woo_settings_list",
            "woocommerce-woo_system_status",
            "word--add_paragraph",
            "word-add_heading",
            "word-add_paragraph",
            "word-add_table",
            "word-copy_document",
            "word-create_custom_style",
            "word-create_document",
            "word-delete_paragraph",
            "word-find_text_in_document",
            "word-format_table",
            "word-format_table_cell_text",
            "word-format_text",
            "word-get_document_info",
            "word-get_document_outline",
            "word-get_document_text",
            "word-get_paragraph_text_from_document",
            "word-highlight_table_header",
            "word-insert_header_near_text",
            "word-insert_line_or_paragraph_near_text",
            "word-list_available_documents",
            "word-search_and_replace",
            "word-set_table_alignment_all",
            "word-set_table_cell_alignment",
            "word-set_table_cell_shading",
            "yahoo--finance-get_stock_price_by_date",
            "yahoo-finance-get_financial_statement",
            "yahoo-finance-get_historical_stock_prices",
            "yahoo-finance-get_holder_info",
            "yahoo-finance-get_recommendations",
            "yahoo-finance-get_stock_actions",
            "yahoo-finance-get_stock_info",
            "yahoo-finance-get_stock_price_by_date",
            "yahoo-finance_get_stock_price_by_date",
            "youtube-channels_getChannel",
            "youtube-channels_listVideos",
            "youtube-channels_navigateList",
            "youtube-channels_searchVideos",
            "youtube-playlists_getPlaylist",
            "youtube-playlists_getPlaylistItems",
            "youtube-playlists_searchPlaylists",
            "youtube-transcript-get_transcript",
            "youtube-transcripts_getTranscript",
            "youtube-videos_getVideo",
            "youtube-videos_searchVideos",
        ]

        self.num_tools = len(self.TOOLS_LIST)
        
        # 兼容性字段 (model.py 需要读取)
        self.use_lora = True 
        self.lora_r = 16 # 这里的参数主要用于初始化结构，加载权重时会被覆盖
        self.lora_alpha = 32
        self.lora_dropout = 0.05
        self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        self.use_amp = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Original Base Model")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the saved checkpoint (e.g., checkpoints/final_model)")
    parser.add_argument("--data_path", type=str, default="./data/train_ticket_data.json")
    parser.add_argument("--output_file", type=str, default="eval_result.json")
    return parser.parse_args()

class TORLEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        
        print(f"[Eval] Loading Base Model: {cfg.model_path}")
        print(f"[Eval] Loading LoRA Adapter: {cfg.lora_path}")
        
        # 1. 初始化模型架构
        self.policy = TrainTicketPolicyModel(cfg)
        
        # 2. 加载训练好的 LoRA 权重
        # 注意：TrainTicketPolicyModel 初始化时已经挂载了未训练的 LoRA
        # 我们需要用训练好的权重覆盖它
        from peft import PeftModel
        self.policy.model = PeftModel.from_pretrained(
            self.policy.base_model, 
            cfg.lora_path,
            is_trainable=False
        )
        self.policy.to(self.device)
        self.policy.eval() # 切换到评估模式
        
        self.tokenizer = self.policy.tokenizer
        self.env = TrainTicketMockEnv(cfg)
        
        # 加载数据
        self.dataset = TrainTicketRLDataset(cfg)
        print(f"[Eval] Dataset loaded: {len(self.dataset)} episodes")

    def parse_action(self, text):
        """解析工具调用 (与 Trainer 保持一致)"""
        try:
            if "Tool:" in text:
                segment = text.split("Tool:")[-1]
                parts = segment.split("Args:")
                tool_name = parts[0].strip()
                tool_args = parts[1].strip() if len(parts) > 1 else "{}"
                tool_args = tool_args.split("\n")[0]
                return tool_name, tool_args
            return None, None
        except:
            return None, None

    def run_eval_episode(self, episode_data):
        """运行单个测试用例"""
        prompt = episode_data['prompt']
        ground_truth = episode_data['ground_truth']
        
        self.env.reset({"ground_truth": ground_truth})
        
        full_log = f"User: {prompt}\n"
        step = 0
        done = False
        
        while not done and step < self.cfg.max_steps_per_episode:
            input_text = full_log + "Assistant:"
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.policy.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    tokenizer=self.tokenizer,
                    stop_strings=["Observation:", "\nUser:", "<|im_end|>"],
                    do_sample=False, # 评估时使用 Greedy Search 保证确定性
                    temperature=0.0, 
                    top_k=1
                )
            
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            full_log += f"Assistant: {generated_text}\n"
            
            # 解析与执行
            tool_name, tool_args = self.parse_action(generated_text)
            
            if tool_name:
                obs, _, done, info = self.env.step(tool_name, tool_args)
                full_log += f"Observation: {obs}\n"
            else:
                if "train-ticket-plan.json" in generated_text:
                    done = True
                else:
                    # 如果模型既没调用工具也没结束，可能是输出了一些废话，强制结束
                    if step > 10 and "Tool:" not in generated_text:
                        done = True
            
            step += 1
            
        is_success = self.env.check_success()
        return is_success, full_log

    def evaluate(self):
        print("\n>>> Starting Evaluation <<<")
        success_count = 0
        total = len(self.dataset)
        results = []
        
        # 遍历数据集
        for i in tqdm(range(total)):
            ep_data = self.dataset[i]
            success, log_text = self.run_eval_episode(ep_data)
            
            if success:
                success_count += 1
            
            results.append({
                "episode_idx": i,
                "task": ep_data.get("task_name", "unknown"),
                "success": success,
                "log": log_text
            })
            
        accuracy = success_count / total
        print(f"\n========================================")
        print(f"Final Evaluation Results:")
        print(f"Total Episodes: {total}")
        print(f"Success: {success_count}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"========================================")
        
        # 保存结果
        with open(self.cfg.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Detailed logs saved to {self.cfg.output_file}")

if __name__ == "__main__":
    args = parse_args()
    cfg = EvalConfig(args)
    
    evaluator = TORLEvaluator(cfg)
    evaluator.evaluate()