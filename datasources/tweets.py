from pandas.core.construction import sanitize_array
import requests
import os
import json
import pandas as pd

# To set your environment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAEk1RAEAAAAAzjosLGY5ZBoI2510JOg0eZqgPGA%3DTITTljCJgajS4IbV0sUKVysU8KggUUVMYAEgfbpxOATsQPKKYZ'#os.environ.get("BEARER_TOKEN")


def create_url():
    # Replace with user ID below
    user_id = 1037321378
    return "https://api.twitter.com/2/users/{}/tweets".format(user_id)


def get_params():
    # Tweet fields are adjustable.
    # Options include:
    # attachments, author_id, context_annotations,
    # conversation_id, created_at, entities, geo, id,
    # in_reply_to_user_id, lang, non_public_metrics, organic_metrics,
    # possibly_sensitive, promoted_metrics, public_metrics, referenced_tweets,
    # source, text, and withheld
    return {"tweet.fields": "created_at", "max_results": "5"}#, "pagination_token": "7140dibdnow9c7btw3z3al3eejvt8zgiv6ko889o8zfhu", "max_results": "5"}


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2UserTweetsPython"
    return r


def connect_to_endpoint(url, params):
    response = requests.request("GET", url, auth=bearer_oauth, params=params)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()


def main():
    url = create_url()
    params = get_params()
    json_response = connect_to_endpoint(url, params)
    df = pd.json_normalize(json_response)
    print(df)

    #print(json.dumps(json_response, indent=4, sort_keys=True))


if __name__ == "__main__":
    main()  