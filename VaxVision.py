import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass
import re

os.environ["OPENAI_API_KEY"] = "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"

MAX_MEMORY_LESSONS = 5
TIME_DECAY_LESSONS = 0.995
WARMUP_STEPS = 5

random.seed(42)
np.random.seed(42)

class OpenRouterLLM:
    """Simple wrapper for OpenRouter API."""
    
    def __init__(self, model="qwen/qwen3-30b-a3b:free", temperature=0.7):
        self.model = model
        self.temperature = temperature
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
    
    def generate_text(self, prompt, max_tokens=1024):
        """Generate text based on a prompt."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                data=json.dumps(data)
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"API Error {response.status_code}: {response.text}")
                return f"Error: {response.status_code}"
        except Exception as e:
            print(f"API request error: {e}")
            return f"Error: {e}"
    
    def parse_lessons(self, response):
        """
        Parse lessons from a generated response with improved robustness.
        Returns a list of (lesson_text, importance) tuples.
        """
        lessons = []
        
        
        lesson_pattern = r"Lesson\s+\d+\s*:\s*\((.*?),\s*(0\.\d+)\)"
        matches = re.findall(lesson_pattern, response, re.DOTALL)
        
        if matches:
            for match in matches:
                lesson_text, importance_str = match
                try:
                    importance = float(importance_str)
                    lessons.append((lesson_text.strip(), importance))
                except ValueError:
                    pass
        
        if not lessons:
            lesson_pattern = r"Lesson\s+\d+\s*:\s*(.*?)\s*\(importance\s+score:\s*(0\.\d+)\)"
            matches = re.findall(lesson_pattern, response, re.DOTALL)
            
            if matches:
                for match in matches:
                    lesson_text, importance_str = match
                    try:
                        importance = float(importance_str)
                        lessons.append((lesson_text.strip(), importance))
                    except ValueError:
                        pass
        
        if not lessons:
            lines = response.split('\n')
            for line in lines:
                if "Lesson" in line and any(c.isdigit() for c in line):
                    score_matches = re.findall(r'(0\.\d+)', line)
                    if score_matches:
                        importance = float(score_matches[-1])  # Take the last one as the importance
                        lesson_text = line[:line.rfind(score_matches[-1])].strip()
                        lesson_text = re.sub(r'^Lesson\s+\d+\s*:', '', lesson_text).strip()
                        lessons.append((lesson_text, importance))
        
        if not lessons and response.strip():
            lessons.append((response.strip(), 0.5))
        
        if not lessons:
            print(f"No valid lessons parsed from response:\n{response}")
        
        return lessons
    
    def parse_attitude_distribution(self, response):
        """
        Parse attitude distribution from a response with improved robustness.
        Returns a list of 4 probabilities.
        """
        # 1. First try the standard format: [0.1, 0.2, 0.3, 0.4]
        pattern = r'\[([0-9\.]+),\s*([0-9\.]+),\s*([0-9\.]+),\s*([0-9\.]+)\]'
        match = re.search(pattern, response)
        
        if match:
            try:
                dist = [float(match.group(i)) for i in range(1, 5)]
                # Normalize
                sum_dist = sum(dist)
                if sum_dist > 0:
                    return [d / sum_dist for d in dist]
            except ValueError:
                pass
        
        matches = re.findall(r'(0\.\d+)', response)
        if len(matches) >= 4:
            try:
                dist = [float(matches[i]) for i in range(4)]
                sum_dist = sum(dist)
                if sum_dist > 0:
                    return [d / sum_dist for d in dist]
            except ValueError:
                pass
        
        attitude_match = re.search(r'attitude(?:.*?)([1-4])', response, re.IGNORECASE)
        if attitude_match:
            attitude = int(attitude_match.group(1))
            dist = [0.1, 0.1, 0.1, 0.1]
            dist[attitude-1] = 0.7  
            return dist
        
        if re.search(r'definitely\s+no|absolutely\s+not|strongly\s+against', response, re.IGNORECASE):
            return [0.7, 0.2, 0.05, 0.05]  
        elif re.search(r'probably\s+not|hesitant|skeptical|doubtful', response, re.IGNORECASE):
            return [0.3, 0.5, 0.15, 0.05]  
        elif re.search(r'probably\s+yes|likely|inclined', response, re.IGNORECASE):
            return [0.05, 0.15, 0.5, 0.3]  
        elif re.search(r'definitely\s+yes|absolutely|strongly\s+support', response, re.IGNORECASE):
            return [0.05, 0.05, 0.2, 0.7]  
        
        print(f"Failed to parse attitude distribution from response")
        return [0.25, 0.25, 0.25, 0.25]
    
    def parse_tweet(self, response):
        match = re.search(r'\*(.*?)(?:\*|$)', response)
        if match:
            return match.group(1).strip()
        
        paragraphs = [p.strip() for p in response.split('\n') if p.strip()]
        for p in paragraphs:
            if 30 <= len(p) <= 280:
                return p
        
        if paragraphs:
            shortest = min(paragraphs, key=len)
            if len(shortest) <= 300:  
                return shortest
        
        truncated = response.strip()
        if len(truncated) > 280:
            truncated = truncated[:277] + "..."
        return truncated

@dataclass
class Lesson:
    content: str
    importance: float
    time_created: int
    
    def __str__(self):
        return f"{self.content} (Importance: {self.importance:.2f})"

class Agent:
    
    def __init__(self, id, llm, demographics, temperature=0.7):
        self.id = id
        self.llm = llm
        self.demographics = demographics
        self.temperature = temperature
        
        self.memory = []
        self.tweets = []
        self.attitudes = []
        self.current_attitude = None
    
    def process_step(self, news, policy, disease_risk, tweets, step):
        self._process_news_and_policy(news, policy, disease_risk, step)
        
        self._process_tweets(tweets, step)
        
        self._write_tweet(step)
        
        attitude = self._determine_attitude(step)
        
        return attitude
    
    def _process_news_and_policy(self, news, policy, disease_risk, step):
        news_content = "\n".join([f"News: {n}" for n in news])
        policy_content = f"Policy: {policy}" if policy else ""
        risk_content = f"This week, the government has reported a {disease_risk} level of disease risk."
        
        content = f"{news_content}\n{policy_content}\n{risk_content}"
        
        top_memories = self._get_top_memories()
        memory_content = "\n".join([str(memory) for memory in top_memories])
        
        prompt = self._create_lesson_prompt(content, memory_content)
        
        response = self.llm.generate_text(prompt)
        lessons = self.llm.parse_lessons(response)
        
        for lesson_text, importance in lessons:
            self.memory.append(Lesson(
                content=lesson_text,
                importance=importance,
                time_created=step
            ))
    
    def _process_tweets(self, tweets, step):
        if not tweets:
            return
        
        tweet_content = "\n".join([f"Tweet: {t['content']}" for t in tweets])
        
        top_memories = self._get_top_memories()
        memory_content = "\n".join([str(memory) for memory in top_memories])
        
        prompt = self._create_tweet_lesson_prompt(tweet_content, memory_content)
        
        response = self.llm.generate_text(prompt)
        lessons = self.llm.parse_lessons(response)
        
        for lesson_text, importance in lessons:
            self.memory.append(Lesson(
                content=lesson_text,
                importance=importance,
                time_created=step
            ))
    
    def _write_tweet(self, step):
        top_memories = self._get_top_memories()
        memory_content = "\n".join([str(memory) for memory in top_memories])
        
        prompt = self._create_tweet_writing_prompt(memory_content)
        
        response = self.llm.generate_text(prompt)
        tweet_content = self.llm.parse_tweet(response)
        
        tweet = {
            'agent_id': self.id,
            'content': tweet_content,
            'step': step
        }
        
        self.tweets.append(tweet)
        
        return tweet
    
    def _determine_attitude(self, step):
        top_memories = self._get_top_memories()
        memory_content = "\n".join([str(memory) for memory in top_memories])
        
        prompt = self._create_attitude_prompt(memory_content)
        
        response = self.llm.generate_text(prompt)
        attitude_dist = self.llm.parse_attitude_distribution(response)
        
        modulated_dist = self._modulate_attitude_distribution(attitude_dist)
        
        attitude = np.random.choice([1, 2, 3, 4], p=modulated_dist)
        
        self.attitudes.append(attitude)
        self.current_attitude = attitude
        
        return attitude
    
    def _get_top_memories(self):
        if not self.memory:
            return []
        
        current_time = len(self.attitudes) if self.attitudes else 0
        saliency_scores = []
        
        for lesson in self.memory:
            time_diff = current_time - lesson.time_created
            decay_factor = TIME_DECAY_LESSONS ** time_diff
            saliency = lesson.importance + decay_factor
            saliency_scores.append(saliency)
        
        if saliency_scores:
            min_score = min(saliency_scores)
            max_score = max(saliency_scores)
            if max_score > min_score:
                normalized_scores = [(score - min_score) / (max_score - min_score) 
                                   for score in saliency_scores]
            else:
                normalized_scores = [1.0 for _ in saliency_scores]
        else:
            normalized_scores = []
        
        memory_saliency = list(zip(self.memory, normalized_scores))
        
        memory_saliency.sort(key=lambda x: x[1], reverse=True)
        top_memories = [mem for mem, _ in memory_saliency[:MAX_MEMORY_LESSONS]]
        
        return top_memories
    
    def _modulate_attitude_distribution(self, attitude_dist):
        dist = np.array(attitude_dist)
        
        if np.sum(dist) == 0:
            return [0.25, 0.25, 0.25, 0.25]
        
        if np.sum(dist) != 1.0:
            dist = dist / np.sum(dist)
        
        epsilon = 1e-10
        log_dist = np.log(dist + epsilon)
        scaled_log_dist = log_dist / self.temperature
        
        exp_scaled = np.exp(scaled_log_dist)
        modulated_dist = exp_scaled / np.sum(exp_scaled)
        
        return modulated_dist.tolist()
    
    def _create_lesson_prompt(self, content, memory):
        persona = self._format_demographics()
        
        prompt = f"""Pretend you are a person with the following profile:
{persona}

You have the following important memories and lessons:
{memory}

You read the following news and policies:
{content}

Based on your profile and the information above, summarize 1-2 new lessons you've learned from reading 
this news and policies. Each lesson should be a brief paragraph. Don't repeat your existing memories.

For each lesson, also assign an importance score as a decimal number between 0 and 1, where 1 is extremely important.

Format your response as:
Lesson 1: (lesson text, 0.8)
Lesson 2: (lesson text, 0.7)
"""
        return prompt
    
    def _create_tweet_lesson_prompt(self, tweets, memory):
        persona = self._format_demographics()
        
        prompt = f"""Pretend you are a person with the following profile:
{persona}

You have the following important memories and lessons:
{memory}

You read the following tweets posted by other people:
{tweets}

Based on your profile and the information above, summarize 1-2 new lessons you've learned from reading 
these tweets. Each lesson should be a brief paragraph. Don't repeat your existing memories.

For each lesson, also assign an importance score as a decimal number between 0 and 1, where 1 is extremely important.

Format your response as:
Lesson 1: (lesson text, 0.8)
Lesson 2: (lesson text, 0.7)
"""
        return prompt
    
    def _create_tweet_writing_prompt(self, memory):
        persona = self._format_demographics()
        
        prompt = f"""Pretend you are a person with the following profile:
{persona}

You have the following important memories and lessons:
{memory}

Write a tweet about COVID-19 vaccinations expressing your opinions on this topic. 
Your tweet should reflect your personality, beliefs, and knowledge from your memories.
Begin your tweet with an asterisk (*) and keep it under 280 characters.

For example:
* I believe vaccines are safe and effective. The scientific evidence is clear!

or

* Still skeptical about the COVID vaccine. Need more long-term data before making my decision.
"""
        return prompt
    
    def _create_attitude_prompt(self, memory):
        persona = self._format_demographics()
        
        prompt = f"""Pretend you are a person with the following profile:
    {persona}

    You have the following important memories and lessons:
    {memory}

    Vaccine hesitancy is defined as the delay in acceptance or refusal of vaccines despite the availability of vaccination services. 
    It's influenced by factors such as:
    - Complacency (low perceived risk of disease)
    - Convenience (access barriers)
    - Confidence (trust in vaccine effectiveness and safety)
    - Social and psychological factors (peer influence, political beliefs)

    Based on your profile, memories, and these factors, if a vaccine to prevent the disease were offered to you today, 
    would you choose to get vaccinated? On an integer scale of 1-4:
    1 = You will definitely not get vaccinated
    2 = You will probably not get vaccinated
    3 = You will probably get vaccinated
    4 = You will definitely get vaccinated

    First, provide your reasoning (1-2 paragraphs) reflecting your personality and beliefs. Be stubborn if your character would be, 
    and don't be easily swayed by general information if it contradicts your core beliefs.

    Then, output your answer as a list of four floating-point numbers representing the probabilities for each attitude rating (1-4).
    The probabilities must sum to 1.0. Format the list exactly as follows:
    [0.1, 0.2, 0.3, 0.4]

    Example response:
    Reasoning: I trust the science behind vaccines, and my family has always been pro-vaccination. However, I’ve read about rare side effects, so I’m not 100% certain.
    [0.05, 0.1, 0.35, 0.5]
    """
        return prompt
    
    def _format_demographics(self):
        return "; ".join([f"{key.capitalize()}: {value}" for key, value in self.demographics.items()])

class VACSIMFramework:
    
    def __init__(self, model_name="deepseek/deepseek-v3-base:free", temperature=0.7, num_agents=5):
        self.model_name = model_name
        self.temp_modulation = temperature
        self.num_agents = num_agents
        
        # Initialize LLM
        self.llm = OpenRouterLLM(model=model_name, temperature=0.7)
        
        # Initialize components
        self.agents = []
        self.news_corpus = {
            'pro_vaccine': [],
            'anti_vaccine': [],
            'low_disruption': [],
            'high_disruption': []
        }
        self.social_network = {}
        self.tweet_database = []
        self.disease_risk_data = []
        
        # Initialize state
        self.current_step = 0
        self.policy = None
        self.policy_effort = None
        self.hesitancy_trajectory = []
    
    def initialize_simulation(self):
        self._create_agents()
        self._generate_sample_news()
        self._build_social_network()
        self._generate_disease_risk_data()
        print("Simulation initialized.")
    
    def run_simulation(self, policy_type=None, policy_effort="weak", steps=2):
        self.policy = policy_type
        self.policy_effort = policy_effort
        self.hesitancy_trajectory = []
        
        print(f"Running simulation: policy={policy_type}, effort={policy_effort}, steps={steps}")
        
        for step in range(steps):
            self.current_step = step
            print(f"Step {step+1}/{steps}...")
            
            applied_policy = None
            if step >= WARMUP_STEPS and policy_type is not None:
                applied_policy = self._get_policy_description()
            
            current_risk = self.disease_risk_data[step]
            
            agent_attitudes = []
            for agent in self.agents:
                agent_news = self._recommend_news(agent)
                agent_tweets = self._recommend_tweets(agent)
                
                attitude = agent.process_step(
                    news=agent_news,
                    policy=applied_policy,
                    disease_risk=current_risk,
                    tweets=agent_tweets,
                    step=step
                )
                
                if agent.tweets and len(agent.tweets) > 0:
                    latest_tweet = agent.tweets[-1]
                    if latest_tweet not in self.tweet_database:
                        self.tweet_database.append(latest_tweet)
                
                agent_attitudes.append(attitude)
            
            hesitancy_rate = sum(1 for att in agent_attitudes if att <= 2) / len(agent_attitudes)
            self.hesitancy_trajectory.append(hesitancy_rate)
            
            print(f"Hesitancy rate: {hesitancy_rate:.2f}")
        
        return self.hesitancy_trajectory
    
    def _create_agents(self):
        """Create the agent population."""
        print("Creating agents...")
        self.agents = []
        for i in range(self.num_agents):
            agent = Agent(
                id=i,
                llm=self.llm,
                demographics=self._sample_demographics(),
                temperature=self.temp_modulation
            )
            self.agents.append(agent)
        print(f"Created {len(self.agents)} agents.")
    
    def _sample_demographics(self):
        """Sample a demographic profile for an agent."""
        genders = ["Male", "Female"]
        ages = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
        educations = ["High School", "Some College", "Bachelor's Degree", "Graduate Degree"]
        occupations = ["Healthcare", "Education", "Service", "Professional", "Retired", "Student"]
        political_beliefs = ["Liberal", "Conservative", "Moderate", "Independent"]
        religions = ["Christianity", "Judaism", "Islam", "Hinduism", "Buddhism", "None"]
        
        return {
            "gender": random.choice(genders),
            "age": random.choice(ages),
            "education": random.choice(educations),
            "occupation": random.choice(occupations),
            "political_belief": random.choice(political_beliefs),
            "religion": random.choice(religions)
        }
    
    def _generate_sample_news(self):
        """Generate a small set of sample news articles."""
        print("Generating sample news...")
        
        self.news_corpus['pro_vaccine'] = [
            "Study shows COVID-19 vaccines are 95% effective at preventing severe illness and hospitalization.",
            "Vaccinated individuals show significantly lower transmission rates of COVID-19 according to new research.",
            "Public health experts confirm vaccine side effects are typically mild and short-lived."
        ]
        
        self.news_corpus['anti_vaccine'] = [
            "Some individuals report concerning side effects after receiving COVID-19 vaccines.",
            "Questions raised about long-term studies on new vaccine technologies.",
            "Group calls for more transparency about vaccine approval process."
        ]
        
        self.news_corpus['low_disruption'] = [
            "COVID-19 cases declining in most regions, authorities consider reducing restrictions.",
            "Many businesses returning to normal operations as pandemic threat diminishes.",
            "Hospitals report manageable COVID-19 patient loads as severe cases decrease."
        ]
        
        self.news_corpus['high_disruption'] = [
            "COVID-19 cases surging in multiple regions, overwhelming local healthcare systems.",
            "New restrictions being considered as ICU capacity reaches critical levels.",
            "Schools forced to return to remote learning amid rising infection rates."
        ]
        
        print("Sample news generated.")
    
    def _build_social_network(self):
        """Build a simple social network between agents."""
        print("Building social network...")
        
        self.social_network = {}
        for agent in self.agents:
            num_follows = random.randint(1, min(2, len(self.agents)-1))
            follows = random.sample([a.id for a in self.agents if a.id != agent.id], 
                                   min(num_follows, len(self.agents)-1))
            self.social_network[agent.id] = follows
        
        print("Social network built.")
    
    def _generate_disease_risk_data(self):
        """Generate disease risk data for the simulation."""
        risk_levels = []
        for i in range(20): 
            if i < 5:
                risk = "high"
            elif i < 10:
                risk = random.choice(["high", "medium", "medium"])
            elif i < 15:
                risk = random.choice(["medium", "medium", "low"])
            else:
                risk = random.choice(["medium", "low", "low"])
            risk_levels.append(risk)
        
        self.disease_risk_data = risk_levels
    
    def _recommend_news(self, agent):
        """Recommend news articles to an agent."""
        recommended_news = []
        for category in self.news_corpus.keys():
            if self.news_corpus[category]:
                news = random.choice(self.news_corpus[category])
                recommended_news.append(news)
        
        if len(recommended_news) > 3:
            recommended_news = random.sample(recommended_news, 3)
        
        return recommended_news
    
    def _recommend_tweets(self, agent):
        """Recommend tweets to an agent."""
        if not self.tweet_database:
            return []
        
        follows = self.social_network.get(agent.id, [])
        relevant_tweets = [t for t in self.tweet_database 
                          if t['agent_id'] in follows 
                          and t['step'] < self.current_step]
        
        if len(relevant_tweets) < 2 and self.tweet_database:
            other_tweets = [t for t in self.tweet_database 
                           if t['agent_id'] not in follows 
                           and t['agent_id'] != agent.id
                           and t['step'] < self.current_step]
            if other_tweets:
                relevant_tweets.extend(random.sample(other_tweets, 
                                                   min(2 - len(relevant_tweets), len(other_tweets))))
        
        if len(relevant_tweets) > 2:
            return random.sample(relevant_tweets, 2)
        return relevant_tweets
    
    def _get_policy_description(self):
        """Get the description of the current policy."""
        policies = {
            "financial": {
                "weak": "The state government offers a $10 cash card to adults who receive their first dose of vaccination.",
                "strong": "The state government offers a $50 cash card to adults who receive their first dose of vaccination."
            },
            "ambassador": {
                "weak": "The government launches a community ambassador program where local leaders share information about vaccines.",
                "strong": "The government launches an extensive community ambassador program where trusted local leaders actively promote vaccination and address concerns."
            },
            "mandate": {
                "weak": "The government requires proof of vaccination or weekly testing for certain public employees.",
                "strong": "The government mandates vaccination for all public employees and for entrance to restaurants, gyms, and other indoor venues."
            }
        }
        
        if self.policy in policies and self.policy_effort in policies[self.policy]:
            return policies[self.policy][self.policy_effort]
        return None
    
    def plot_hesitancy_trajectory(self, save_path=None):
        """Plot the hesitancy trajectory over time."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.hesitancy_trajectory)), self.hesitancy_trajectory, 
                marker='o', linestyle='-', color='blue')
        
        if len(self.hesitancy_trajectory) > WARMUP_STEPS:
            plt.axvline(x=WARMUP_STEPS, color='r', linestyle='--', 
                      label=f'Policy Applied (Week {WARMUP_STEPS+1})')
        
        plt.title(f'Vaccine Hesitancy Over Time\nPolicy: {self.policy}, Effort: {self.policy_effort}')
        plt.xlabel('Simulation Week')
        plt.ylabel('Hesitancy Rate')
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def analyze_results(self):
        if not self.hesitancy_trajectory:
            return "No simulation results to analyze."
        
        initial_hesitancy = self.hesitancy_trajectory[0]
        final_hesitancy = self.hesitancy_trajectory[-1]
        change = final_hesitancy - initial_hesitancy
        
        pro_vaccine = [a for a in self.agents if a.current_attitude >= 3]
        hesitant = [a for a in self.agents if a.current_attitude <= 2]
        
        analysis = {
            "initial_hesitancy": initial_hesitancy,
            "final_hesitancy": final_hesitancy,
            "hesitancy_change": change,
            "pro_vaccine_count": len(pro_vaccine),
            "hesitant_count": len(hesitant),
            "agent_details": []
        }
        
        for agent in self.agents:
            if len(agent.attitudes) >= 2:
                agent_info = {
                    "id": agent.id,
                    "demographics": agent.demographics,
                    "initial_attitude": agent.attitudes[0],
                    "final_attitude": agent.attitudes[-1],
                    "attitude_change": agent.attitudes[-1] - agent.attitudes[0],
                    "sample_tweet": agent.tweets[-1]["content"] if agent.tweets else "No tweets"
                }
                analysis["agent_details"].append(agent_info)
        
        return analysis

    def save_results(self, output_dir="./results"):
        os.makedirs(output_dir, exist_ok=True)
        
        trajectory_df = pd.DataFrame({
            "step": range(len(self.hesitancy_trajectory)),
            "hesitancy": self.hesitancy_trajectory
        })
        trajectory_df.to_csv(os.path.join(output_dir, "hesitancy_trajectory.csv"), index=False)
        
        attitudes = []
        for agent in self.agents:
            for step, attitude in enumerate(agent.attitudes):
                attitudes.append({
                    "agent_id": agent.id,
                    "step": step,
                    "attitude": attitude
                })
        
        attitudes_df = pd.DataFrame(attitudes)
        attitudes_df.to_csv(os.path.join(output_dir, "agent_attitudes.csv"), index=False)
        
        demographics = []
        for agent in self.agents:
            demo = {"agent_id": agent.id}
            demo.update(agent.demographics)
            demographics.append(demo)
        
        demographics_df = pd.DataFrame(demographics)
        demographics_df.to_csv(os.path.join(output_dir, "agent_demographics.csv"), index=False)
        
        tweets = []
        for agent in self.agents:
            for tweet in agent.tweets:
                tweet_data = {
                    "agent_id": tweet["agent_id"],
                    "step": tweet["step"],
                    "content": tweet["content"]
                }
                tweets.append(tweet_data)
        
        tweets_df = pd.DataFrame(tweets)
        tweets_df.to_csv(os.path.join(output_dir, "agent_tweets.csv"), index=False)
        
        self.plot_hesitancy_trajectory(save_path=os.path.join(output_dir, "hesitancy_plot.png"))
        
        analysis = self.analyze_results()
        with open(os.path.join(output_dir, "analysis_summary.json"), "w") as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Results saved to {output_dir}")

def run_simple_demo(model_name="meta-llama/llama-3-8b-instruct"):
    os.makedirs("./results", exist_ok=True)
    
    print("Initializing simulation...")
    framework = VACSIMFramework(
        model_name=model_name,
        temperature=0.7,
        num_agents=10  
    )
    
    framework.initialize_simulation()
    
    try:
        print("Running simulation...")
        framework.run_simulation(
            policy_type="financial",
            policy_effort="strong",
            steps=20
        )
        
        framework.save_results()
        
        analysis = framework.analyze_results()
        print("\nSimulation Summary:")
        print(f"Initial hesitancy rate: {analysis['initial_hesitancy']:.2f}")
        print(f"Final hesitancy rate: {analysis['final_hesitancy']:.2f}")
        print(f"Change in hesitancy: {analysis['hesitancy_change']:.2f}")
        print(f"Pro-vaccine agents: {analysis['pro_vaccine_count']}/{framework.num_agents}")
        print(f"Hesitant agents: {analysis['hesitant_count']}/{framework.num_agents}")
        print("\nDetailed results saved to ./results/")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    
    return framework

if __name__ == "__main__":
    model = "qwen/qwen3-30b-a3b:free"
    print(f"Running simulation with model: {model}")
    framework = run_simple_demo(model)