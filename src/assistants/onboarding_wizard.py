from src.assistants.base_wizard import BaseWizard

from src.tools import fetch_user_activities, update_activity, create_activity


class OnboardingWizard(BaseWizard):
    @property
    def safe_tools(self):
        return [fetch_user_activities]

    @property
    def sensitive_tools(self):
        return [update_activity, create_activity]

    @property
    def name(self):
        return "onboarding_wizard"


def main():
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI
    from src.prompts.yaml_prompt_loader import YamlPromptLoader

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
    from nemoguardrails import RailsConfig

    config = RailsConfig.from_path("./config")
    guardrails = RunnableRails(config)

    llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    prompt = ChatPromptTemplate.from_messages([("system", "You respond concisely."), ("user", "{input}")])
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    chain_w_rails = guardrails | chain

    result = chain_w_rails.invoke({"input": "Wassup?"})

    intermediate_result = chain.invoke({"input": "Wassup?"})
    print(intermediate_result)  # Check the output before guardrails
    result = chain_w_rails.invoke({"input": "Wassup?"})
    print(result)

    # Test the flow directly
    flow_result = guardrails.run_flow("greeting", {"input": "Wassup?"})
    print(flow_result)

    print("done?")


if __name__ == "__main__":
    main()
