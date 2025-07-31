# main.py - Structural Similarity Analysis Based on Gentner (1983)

import pandas as pd
import requests
import json
import time
import os
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
from prompts import STRUCTURAL_SIMILARITY_PROMPT
import numpy as np

# --- Configuration ---
OPENROUTER_CONFIG = {
    'api_key': 'Your KEY here', # Add your own key form openrouter.ai here
    'base_url': 'https://openrouter.ai/api/v1',
    'model_name': 'deepseek/deepseek-chat-v3-0324',
    'max_tokens': 2000,  # Reduced since we're only analyzing one dimension
    'temperature': 0.1,
    'timeout': 120
}

INPUT_CSV = '300_final.csv'
OUTPUT_DIR = 'structural_similarity_results'
OUTPUT_CSV = 'structural_similarity_analysis.csv'
CHECKPOINT_CSV = 'structural_similarity_checkpoint.csv'

# --- Helper Functions ---

def call_ai_model(prompt):
    """
    Sends a prompt to the OpenRouter API and returns the parsed JSON response.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_CONFIG['api_key']}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": OPENROUTER_CONFIG['model_name'],
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "max_tokens": OPENROUTER_CONFIG['max_tokens'],
        "temperature": OPENROUTER_CONFIG['temperature']
    }
    
    try:
        response = requests.post(
            OPENROUTER_CONFIG['base_url'] + "/chat/completions",
            headers=headers,
            json=data,
            timeout=OPENROUTER_CONFIG['timeout']
        )
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        return json.loads(content)
        
    except Exception as e:
        print(f"  -> API Error: {e}")
        return None

def extract_structural_features(ai_response, row_id):
    """
    Extracts structural similarity features from AI response.
    """
    print(f"    -> Extracting structural similarity features for row {row_id}...")
    
    if not ai_response:
        print(f"    -> No AI response for row {row_id}")
        return create_empty_features()
    
    try:
        features = {}
        
        # Extract main features
        features['structural_similarity_score'] = ai_response.get('structural_similarity_score', 0)
        features['analysis_category'] = ai_response.get('analysis_category', 'unknown')
        features['explanation'] = ai_response.get('explanation', '')
        
        # Extract lists
        relational_expressions = ai_response.get('relational_expressions', [])
        attribute_expressions = ai_response.get('attribute_expressions', [])
        key_evidence = ai_response.get('key_evidence', [])
        
        features['relational_count'] = len(relational_expressions)
        features['attribute_count'] = len(attribute_expressions)
        features['relational_expressions'] = '; '.join(relational_expressions)
        features['attribute_expressions'] = '; '.join(attribute_expressions)
        features['key_evidence'] = '; '.join(key_evidence)
        
        # Create binary variable for statistical testing
        features['is_structural'] = 1 if features['analysis_category'] == 'structural' else 0
        features['is_mixed'] = 1 if features['analysis_category'] == 'mixed' else 0
        features['is_non_structural'] = 1 if features['analysis_category'] == 'non-structural' else 0
        
        # Calculate relational ratio
        total_expressions = features['relational_count'] + features['attribute_count']
        if total_expressions > 0:
            features['relational_ratio'] = features['relational_count'] / total_expressions
        else:
            features['relational_ratio'] = 0
        
        # Add processing status
        features['analysis_status'] = 'success'
        features['processed_timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"    -> ✅ Successfully extracted structural similarity features")
        print(f"    -> Score: {features['structural_similarity_score']}, Category: {features['analysis_category']}")
        return features
        
    except Exception as e:
        print(f"    -> ❌ Error extracting features: {e}")
        return create_empty_features_with_error(str(e))

def create_empty_features():
    """
    Creates a dictionary with all expected features set to default values.
    """
    return {
        'structural_similarity_score': 0,
        'analysis_category': 'unknown',
        'explanation': '',
        'relational_count': 0,
        'attribute_count': 0,
        'relational_expressions': '',
        'attribute_expressions': '',
        'key_evidence': '',
        'is_structural': 0,
        'is_mixed': 0,
        'is_non_structural': 0,
        'relational_ratio': 0,
        'analysis_status': 'failed',
        'processed_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def create_empty_features_with_error(error_msg):
    """
    Creates empty features with error information.
    """
    features = create_empty_features()
    features['error_message'] = error_msg
    return features

def save_progress(df, output_path, row_index):
    """
    Saves progress incrementally.
    """
    try:
        df.to_csv(output_path, index=False, sep=';')
        if row_index % 10 == 0:
            print(f"    -> 💾 Progress saved at row {row_index}")
    except Exception as e:
        print(f"    -> ⚠️ Failed to save progress: {e}")

def check_needs_processing(row):
    """
    Check if a row needs processing based on explanation field.
    Returns True if the row needs processing.
    """
    explanation = row.get('explanation')
    
    # Check if it's NaN, None, or empty
    if pd.isna(explanation) or explanation is None:
        return True
    
    # Check if it's a string with less than 10 characters
    if isinstance(explanation, str) and len(explanation.strip()) < 10:
        return True
    
    return False

def perform_structural_similarity_analysis(df):
    """
    Performs statistical analysis on structural similarity data.
    """
    print("\n" + "="*60)
    print("🔬 STRUCTURAL SIMILARITY ANALYSIS (Gentner, 1983)")
    print("="*60)
    
    # Filter out rows where analysis failed
    successful = df[df['analysis_status'] == 'success']
    print(f"\n📊 Analyzing {len(successful)} successful analyses out of {len(df)} total rows")
    
    if len(successful) < 10:
        print("⚠️ Too few successful analyses for reliable statistics")
        return
    
    # Group by expertise
    advanced = successful[successful['expertise'] == 'Advanced']
    novice = successful[successful['expertise'] == 'Novice']
    
    print(f"\n📈 GROUP SIZES:")
    print(f"Advanced players: {len(advanced)}")
    print(f"Novice players: {len(novice)}")
    
    # 1. Category Distribution Analysis
    print(f"\n🔍 STRUCTURAL SIMILARITY CATEGORIES:")
    print("-" * 50)
    
    # Create contingency table
    contingency_table = pd.crosstab(successful['expertise'], successful['analysis_category'])
    percentages = pd.crosstab(successful['expertise'], successful['analysis_category'], normalize='index') * 100
    
    print("\nContingency Table:")
    print(contingency_table)
    print("\nPercentages by Expertise:")
    print(percentages.round(1))
    
    # Chi-square test
    if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f"\nChi-Square Test: χ² = {chi2:.4f}, p = {p:.4f}, df = {dof}")
        
        if p < 0.001:
            print("Result: *** HIGHLY SIGNIFICANT ***")
        elif p < 0.01:
            print("Result: ** SIGNIFICANT **")
        elif p < 0.05:
            print("Result: * SIGNIFICANT *")
        else:
            print("Result: Not significant")
    
    # 2. Structural Similarity Score Analysis
    print(f"\n📊 STRUCTURAL SIMILARITY SCORES:")
    print("-" * 50)
    
    # Descriptive statistics
    print("\nDescriptive Statistics:")
    score_stats = successful.groupby('expertise')['structural_similarity_score'].agg([
        'count', 'mean', 'std', 'min', 'max', 'median'
    ]).round(2)
    print(score_stats)
    
    # T-test for scores
    if len(advanced) > 1 and len(novice) > 1:
        t_stat, p_value = ttest_ind(
            advanced['structural_similarity_score'], 
            novice['structural_similarity_score']
        )
        print(f"\nT-test (Independent Samples):")
        print(f"t = {t_stat:.4f}, p = {p_value:.4f}")
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(advanced)-1)*advanced['structural_similarity_score'].std()**2 + 
                             (len(novice)-1)*novice['structural_similarity_score'].std()**2) / 
                            (len(advanced) + len(novice) - 2))
        cohen_d = (advanced['structural_similarity_score'].mean() - 
                  novice['structural_similarity_score'].mean()) / pooled_std
        print(f"Cohen's d = {cohen_d:.3f} ({'large' if abs(cohen_d) > 0.8 else 'medium' if abs(cohen_d) > 0.5 else 'small'} effect)")
        
        # Mann-Whitney U test (non-parametric alternative)
        u_stat, u_p = mannwhitneyu(
            advanced['structural_similarity_score'], 
            novice['structural_similarity_score']
        )
        print(f"\nMann-Whitney U test:")
        print(f"U = {u_stat:.1f}, p = {u_p:.4f}")
    
    # 3. Relational vs Attribute Expression Analysis
    print(f"\n🔗 RELATIONAL VS ATTRIBUTE EXPRESSIONS:")
    print("-" * 50)
    
    relational_stats = successful.groupby('expertise')[['relational_count', 'attribute_count', 'relational_ratio']].agg(['mean', 'std']).round(2)
    print(relational_stats)
    
    # 4. Binary Analysis - Is Structural?
    print(f"\n🎯 BINARY ANALYSIS - STRUCTURAL THINKING:")
    print("-" * 50)
    
    structural_pct = successful.groupby('expertise')['is_structural'].mean() * 100
    print(f"\nPercentage showing structural thinking:")
    print(structural_pct.round(1))
    
    # Create 2x2 contingency table for is_structural
    structural_contingency = pd.crosstab(successful['expertise'], successful['is_structural'])
    if structural_contingency.shape == (2, 2):
        chi2, p, dof, expected = chi2_contingency(structural_contingency)
        print(f"\nChi-Square Test for Structural Thinking:")
        print(f"χ² = {chi2:.4f}, p = {p:.4f}")
    
    # 5. Summary Report
    print(f"\n🏆 SUMMARY OF FINDINGS:")
    print("="*60)
    
    if len(advanced) > 0 and len(novice) > 0:
        adv_mean = advanced['structural_similarity_score'].mean()
        nov_mean = novice['structural_similarity_score'].mean()
        
        print(f"1. Advanced players show {'higher' if adv_mean > nov_mean else 'lower'} structural similarity")
        print(f"   (M = {adv_mean:.2f} vs M = {nov_mean:.2f})")
        
        adv_structural_pct = (advanced['is_structural'].sum() / len(advanced)) * 100
        nov_structural_pct = (novice['is_structural'].sum() / len(novice)) * 100
        
        print(f"\n2. {adv_structural_pct:.1f}% of advanced players show structural thinking")
        print(f"   vs {nov_structural_pct:.1f}% of novice players")
        
        print(f"\n3. Advanced players use {advanced['relational_count'].mean():.1f} relational expressions on average")
        print(f"   vs {novice['relational_count'].mean():.1f} for novice players")

def main():
    # Setup
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Define paths
    checkpoint_path = os.path.join(OUTPUT_DIR, CHECKPOINT_CSV)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV)
    
    # Show what files exist
    print(f"\n📁 FILE STATUS CHECK:")
    print(f"Original CSV: '{INPUT_CSV}' - {'✅ EXISTS' if os.path.exists(INPUT_CSV) else '❌ MISSING'}")
    print(f"Output CSV: '{output_path}' - {'✅ EXISTS' if os.path.exists(output_path) else '❌ NOT FOUND'}")
    print(f"Checkpoint CSV: '{checkpoint_path}' - {'✅ EXISTS' if os.path.exists(checkpoint_path) else '❌ NOT FOUND'}")
    
    # Try to load existing processed data first
    df = None
    
    # First, try to load the main output file
    if os.path.exists(output_path):
        try:
            df = pd.read_csv(output_path, sep=';')
            print(f"📁 Loaded existing results from '{output_path}' with {len(df)} rows")
        except Exception as e:
            print(f"⚠️ Failed to load existing results: {e}")
    
    # If main output doesn't exist, try checkpoint
    if df is None and os.path.exists(checkpoint_path):
        try:
            df = pd.read_csv(checkpoint_path, sep=';')
            print(f"📁 Loaded checkpoint from '{checkpoint_path}' with {len(df)} rows")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
    
    # If no processed data exists, load the original CSV
    if df is None:
        try:
            df = pd.read_csv(INPUT_CSV, sep=';')
            print(f"✅ Loaded original data from '{INPUT_CSV}' with {len(df)} rows")
        except FileNotFoundError:
            print(f"❌ Error: Input file not found at '{INPUT_CSV}'")
            return
    
    # Clean up the data
    df.dropna(subset=['answer'], inplace=True)
    print(f"📊 After removing empty answers: {len(df)} rows")
    print(f"📊 Expertise distribution: {df['expertise'].value_counts().to_dict()}")
    
    # Initialize results columns in the dataframe if not present
    feature_columns = list(create_empty_features().keys())
    for col in feature_columns:
        if col not in df.columns:
            df[col] = None
    
    # Check how many rows need processing
    rows_needing_processing = df.apply(check_needs_processing, axis=1)
    total_to_process = rows_needing_processing.sum()
    
    print(f"\n🔍 Checking explanation field...")
    print(f"📊 Total rows: {len(df)}")
    print(f"🔄 Rows needing processing: {total_to_process}")
    print(f"✅ Rows already processed: {len(df) - total_to_process}")
    
    if total_to_process == 0:
        print("\n🎉 All rows already processed!")
        print("Running structural similarity analysis on existing data...")
        perform_structural_similarity_analysis(df)
        return
    
    print(f"\n🔍 Starting Structural Similarity Analysis (Gentner, 1983)...")
    print("=" * 60)
    
    processed_count = 0
    
    for index, row in df.iterrows():
        # Check if this row needs processing
        if not check_needs_processing(row):
            print(f"  ✅ Row {index + 1}/{len(df)} already processed, skipping...")
            continue
            
        if not isinstance(row['answer'], str) or not row['answer'].strip():
            print(f"  ⏭️ Row {index + 1}/{len(df)} - empty answer, filling with defaults...")
            empty_features = create_empty_features()
            empty_features['analysis_status'] = 'empty_answer'
            for feature, value in empty_features.items():
                df.at[index, feature] = value
            continue

        print(f"  🔍 Analyzing row {index + 1}/{len(df)} ({row['expertise']})...")
        
        # Create the structural similarity analysis prompt
        prompt = STRUCTURAL_SIMILARITY_PROMPT.format(
            player_type=row['expertise'],
            response_text=row['answer']
        )
        
        # Get AI analysis
        ai_response = call_ai_model(prompt)
        
        # Extract and save features
        features = extract_structural_features(ai_response, index + 1)
        
        # Update the dataframe with extracted features
        for feature, value in features.items():
            df.at[index, feature] = value
        
        processed_count += 1
        
        # Save progress every 5 rows
        if processed_count % 5 == 0:
            save_progress(df, output_path, processed_count)
            # Also save checkpoint
            try:
                df.to_csv(checkpoint_path, index=False, sep=';')
                print(f"    -> 📂 Checkpoint saved")
            except Exception as e:
                print(f"    -> ⚠️ Failed to save checkpoint: {e}")
        
        # Rate limiting
        time.sleep(5)

    # Final save
    print(f"\n💾 Saving final results...")
    try:
        df.to_csv(output_path, index=False, sep=';')
        print(f"✅ Results saved to '{output_path}'")
        
        # Also save as backup
        backup_path = output_path.replace('.csv', '_backup.csv')
        df.to_csv(backup_path, index=False, sep=';')
        print(f"💾 Backup saved to '{backup_path}'")
        
        # Update checkpoint with final results
        df.to_csv(checkpoint_path, index=False, sep=';')
        print(f"📂 Checkpoint updated with final results")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return
    
    # Perform statistical analysis
    try:
        perform_structural_similarity_analysis(df)
    except Exception as e:
        print(f"⚠️ Error in statistical analysis: {e}")
    
    print(f"\n🎉 Analysis Complete!")
    print(f"📊 Processed {processed_count} rows")
    print(f"📁 Results: {output_path}")
    print(f"🔬 Check console output above for statistical results")

if __name__ == '__main__':
    main()