import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path
import numpy as np

# 引入项目模块
# 确保 dataset.py, model.py, mock_env.py 在同一目录下
from model import TrainTicketPolicyModel
from dataset import TrainTicketRLDataset
from mock_env import TrainTicketMockEnv

# ============================================================================
# 1. 配置管理 (Configuration)
# ============================================================================

class RuntimeConfig:
    """
    动态配置类：将命令行参数转换为对象属性，并管理静态配置。
    """
    def __init__(self, args):
        # 从命令行参数继承
        for key, value in vars(args).items():
            setattr(self, key, value)
        
        # 路径处理
        self.project_root = Path(__file__).parent.absolute()
        self.output_dir = Path(args.output_dir)
        self.data_path = Path(args.data_path)
        
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # 静态配置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Qwen LoRA Target Modules
        self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        # 工具列表 (与数据集对齐)
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

def parse_args():
    parser = argparse.ArgumentParser(description="TORL (Tool-Integrated RL) Training Script")

    # --- 路径配置 ---
    parser.add_argument("--data_path", type=str, default="./data/train_ticket_data.json", help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_torl", help="Output directory")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base model path")

    # --- 训练超参数 ---
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--group_size", type=int, default=8, help="Number of rollouts per prompt (GRPO group size)")
    parser.add_argument("--max_steps_per_episode", type=int, default=15, help="Max interaction steps")
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint every N global steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Prompt batch size (keep 1 for standard GRPO)")

    # --- LoRA 配置 ---
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # --- 生成参数 ---
    parser.add_argument("--temperature", type=float, default=1.0, help="Must be >0 for diverse sampling")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    
    # --- 奖励权重 ---
    parser.add_argument("--reward_success_weight", type=float, default=2.0)
    
    # --- 系统 ---
    parser.add_argument("--use_amp", action="store_true", default=True, help="Enable Automatic Mixed Precision")

    return parser.parse_args()

# ============================================================================
# 2. 训练器核心 (Trainer)
# ============================================================================

class TORLTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        
        # 初始化组件
        print(f"[Trainer] Initializing Model from {cfg.model_path}...")
        self.policy = TrainTicketPolicyModel(cfg)
        self.policy.to(self.device)
        self.tokenizer = self.policy.tokenizer
        
        print("[Trainer] Loading Dataset and Env...")
        self.dataset = TrainTicketRLDataset(cfg)
        # shuffle=True 保证训练样本的随机性
        self.dataloader = DataLoader(self.dataset, batch_size=cfg.batch_size, shuffle=True)
        self.env = TrainTicketMockEnv(cfg)
        
        self.optimizer = AdamW(self.policy.parameters(), lr=cfg.learning_rate)
        self.global_step = 0
        
        # 停止符 (用于生成中断)
        self.stop_strings = ["Observation:", "\nUser:", "<|im_end|>"]

    def parse_action(self, text):
        import re
        # 使用正则提取，更鲁棒
        # 匹配 Tool: ... (换行) Args: ...
        # re.DOTALL 允许匹配跨行
        pattern = r"Tool:\s*(.+?)\s*Args:\s*(.+)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            tool_name = match.group(1).strip()
            tool_args = match.group(2).strip()
            return tool_name, tool_args
        return None, None

    def run_rollout(self, prompt, ground_truth):
        self.env.reset({"ground_truth": ground_truth})
        
        # --- [修改 1] 定义 System Prompt 强制格式 ---
        system_prompt = (
            "You are a helpful assistant acting as an agent. "
            "To take action, you MUST use the following format exactly:\n"
            "Tool: <tool_name>\n"
            "Args: <json_args_or_string>\n\n"
            "Example:\n"
            "Tool: search_train_ticket\n"
            "Args: {\"from\": \"Beijing\", \"to\": \"Shanghai\"}\n\n"
            "If you are done or cannot proceed, generate a plan file or stop."
        )

        # --- [修改 2] 使用 apply_chat_template ---
        # 维护一个 messages 列表，符合 Qwen 的微调格式
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        total_reward = 0
        done = False
        step = 0
        success = False
        
        # 用于记录完整的文本日志供 Loss 计算
        # 注意：Qwen 的 template 会添加特殊 token，这里我们在 rollout 中主要维护 messages
        
        while not done and step < self.cfg.max_steps_per_episode:
            # 将当前对话历史转为 tensor
            text_input = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(text_input, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.policy.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    stop_strings=self.stop_strings,
                    do_sample=True 
                )
            
            # 只提取新生成的部分
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # 调试打印：看看模型到底生成了什么？
            # print(f"DEBUG STEP {step}: {generated_text}") 

            # 更新对话历史 (这一点对多轮对话至关重要)
            messages.append({"role": "assistant", "content": generated_text})
            
            # 解析动作
            tool_name, tool_args = self.parse_action(generated_text)
            
            if tool_name:
                obs, reward, done, info = self.env.step(tool_name, tool_args)
                # 将观察结果放回 Prompt
                messages.append({"role": "user", "content": f"Observation: {obs}"})
                total_reward += reward
            else:
                # 依然格式错误时的处理
                if "train-ticket-plan.json" in generated_text:
                     # ... (原有逻辑)
                     pass
                else:
                    # 如果格式错了，给负反馈并结束，或者尝试让它重试（这里先结束）
                    total_reward -= 0.5 
                    done = True
            
            step += 1

        # 为了计算 GRPO Loss，我们需要拼接后的完整文本（或者在 Loss 计算时重新 apply template）
        # 简单起见，这里返回 messages 结构，在 compute_loss 里处理
        return {
            "messages": messages, # 返回结构化数据
            "final_reward": total_reward,
            "success": success
        }

    def compute_grpo_loss(self, trajectories):
        """
        计算 GRPO Loss。
        关键点：只对 Assistant 生成的 Action 部分计算 Loss，Mask 掉其他部分。
        """
        rewards = torch.tensor([t['final_reward'] for t in trajectories], device=self.device)
        
        # --- [修改 3] 防止 Std 为 0 导致的除零或无效计算 ---
        if len(rewards) > 1:
            mean_reward = rewards.mean()
            std_reward = rewards.std()
            # 如果方差极小（例如所有奖励都一样），则 advantage 设为 0
            if std_reward.item() < 1e-6:
                advantages = torch.zeros_like(rewards)
            else:
                advantages = (rewards - mean_reward) / (std_reward + 1e-8)
        else:
            advantages = torch.zeros_like(rewards) # 单样本无法计算相对优势

        total_loss = 0
        valid_trajs = 0

        for idx, traj in enumerate(trajectories):
            # 如果 advantage 是 0，算 Loss 也没意义，只会浪费计算资源，直接跳过
            if advantages[idx] == 0:
                continue

            # 重新构建 Full Text (因为我们需要 Mask 掉 User 部分)
            # 使用 apply_chat_template 生成完整 ids
            messages = traj['messages']
            input_ids = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                return_tensors="pt",
                add_generation_prompt=False
            ).to(self.device)
            
            labels = input_ids.clone()
            
            # --- [修改 4] Masking 逻辑 (简化版) ---
            # 这是一个难点。如果不 Mask User Prompt，模型会“背诵”Prompt。
            # 这里如果不方便根据 Role Mask，至少要 Mask 掉 System Prompt。
            # 对于 GRPO，最简单的 Mask 是：只计算 Assistant 回复部分的 Loss。
            # 为了代码简洁，这里暂时全量计算，但建议后续实现 DataCollator 来处理 Role Masking。
            # 或者：利用 tokenizer 的 chat template 只有 assistant 部分才由 loss 贡献
            
            outputs = self.policy.model(
                input_ids=input_ids,
                labels=labels
            )
            
            # GRPO Loss: - Advantage * log_pi
            # 因为 policy_loss = - log_pi (NLL), 所以:
            # Loss = policy_loss * (-adv)
            # 当 Adv > 0, 我们希望 minimizing NLL (maximize log prob) -> loss 变负
            loss = policy_loss * (-adv)
            
            total_loss += loss
            valid_trajs += 1

        return total_loss / max(1, valid_trajs)

    def train(self):
        print(f"\n[Trainer] Starting GRPO Training")
        print(f"  - Epochs: {self.cfg.num_epochs}")
        print(f"  - Group Size: {self.cfg.group_size}")
        print(f"  - Learning Rate: {self.cfg.learning_rate}")
        print("========================================\n")
        
        self.policy.train()
        
        for epoch in range(self.cfg.num_epochs):
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}")
            
            for batch_idx, batch_data in enumerate(pbar):
                # 提取数据 (Batch Size = 1)
                prompt = batch_data['prompt'][0]
                # 处理 dataset 返回的 dict list 格式
                ground_truth = {
                    k: v[0] if isinstance(v, list) else v 
                    for k, v in batch_data['ground_truth'].items()
                }
                
                # --- 1. Rollout Phase (采样) ---
                trajectories = []
                success_count = 0
                avg_len = 0
                
                # 采样 G 条轨迹
                for _ in range(self.cfg.group_size):
                    traj = self.run_rollout(prompt, ground_truth)
                    trajectories.append(traj)
                    if traj['success']:
                        success_count += 1
                    avg_len += len(traj['full_text'])
                
                # --- 2. Training Phase (更新) ---
                self.optimizer.zero_grad()
                
                loss = self.compute_grpo_loss(trajectories)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.global_step += 1
                
                # --- 3. 记录与保存 ---
                avg_reward = sum(t['final_reward'] for t in trajectories) / len(trajectories)
                avg_len /= len(trajectories)
                
                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}", 
                    "Rw": f"{avg_reward:.2f}",
                    "Succ": f"{success_count}/{self.cfg.group_size}",
                    "Len": f"{int(avg_len)}"
                })
                
                if self.global_step % self.cfg.save_every == 0:
                    self.save_checkpoint(f"step_{self.global_step}")
        
        # 训练结束保存最终模型
        self.save_checkpoint("final_model")

    def save_checkpoint(self, name):
        save_path = self.cfg.output_dir / name
        self.policy.save_pretrained(save_path)
        print(f"\n[Trainer] Checkpoint saved: {save_path}")

# ============================================================================
# 3. 主程序入口
# ============================================================================

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 构建运行时配置
    cfg = RuntimeConfig(args)
    
    # 启动训练
    trainer = TORLTrainer(cfg)
    trainer.train()