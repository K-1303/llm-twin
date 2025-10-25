import requests
from loguru import logger
from typing import Dict, Optional
from llm_engineering.domain.documents import PostDocument
import re
from bson import Binary
import uuid
from llm_engineering.settings import settings


class TwitterCrawler:

    model = PostDocument
    
    def __init__(self):
        """
        Initialize crawler with ScrapingDog API key.
        
        """
        self.api_key = settings.SCRAPINGDOG_API_KEY
        self.base_url = "https://api.scrapingdog.com/x/post"
    
    def extract(self, link: str, **kwargs) -> None:
        """Extract a single tweet from a Twitter/X post URL."""
        try:
            logger.info(f"Extracting tweet from: {link}")
            
            tweet_data = self._scrape_tweet(link)
            
            if tweet_data:
                tweet_data["source_link"] = link
                
                user = kwargs.get("user")
                if user:
                    if isinstance(user.id, uuid.UUID):
                        user_id = Binary.from_uuid(user.id)
                    else:
                        user_id = user.id


                    post = PostDocument(
                        platform="twitter",
                        content=tweet_data,
                        link=link,
                        author_id=user_id,  
                        author_full_name=user.full_name,
                        image=tweet_data.get("image") if tweet_data.get("image") else None)
                    
                    existing_post = self.model.find(link=link)
                    
                    if not existing_post:
                        post.save()
                        logger.info(f"Tweet saved to database")
                    else:
                        logger.info(f"Tweet already exists in database: {link}")
                else:
                    logger.warning("No user object provided, skipping database insert")
            else:
                logger.warning(f"No tweet content found for {link}")

        except Exception as e:
            logger.error(f"Error while scraping tweet {link}: {str(e)}", exc_info=True)
            raise
    
    def _scrape_tweet(self, url: str) -> Optional[Dict[str, str]]:
        """
        Scrape tweet using ScrapingDog API.
        
        Args:
            url: Twitter/X post URL
            
        Returns:
            Dictionary with tweet data or None if failed
        """
        try:
            tweet_id = self._extract_tweet_id(url)
            if not tweet_id:
                logger.error(f"Could not extract tweet ID from URL: {url}")
                return None
            
            params = {
                'api_key': self.api_key,
                'tweetId': tweet_id
            }
            
            logger.debug(f"Calling ScrapingDog API for tweet: {tweet_id}")
            response = requests.get(self.base_url, params=params, timeout=300)

            
            if response.status_code != 200:
                logger.error(f"ScrapingDog API error {response.status_code}: {response.text}")
                return None
            
            data = response.json()
       
            if not data or 'error' in data:
                logger.error(f"Failed to fetch tweet: {data.get('error', 'Unknown error')}")
                return None
            
            tweet_data = self._parse_scrapingdog_response(data)
            
            return tweet_data
            
        except Exception as e:
            logger.error(f"Error in _scrape_tweet: {str(e)}", exc_info=True)
            return None
    
    def _extract_tweet_id(self, url: str) -> Optional[str]:
        """Extract tweet ID from Twitter URL."""
        try:
            match = re.search(r'/status/(\d+)', url)
            if match:
                return match.group(1)
            return None
        except Exception as e:
            logger.error(f"Error extracting tweet ID: {str(e)}")
            return None
    
    def _parse_scrapingdog_response(self, data: dict) -> Dict[str, str]:
        """
        Parse ScrapingDog API response for tweet content with media.
        
        Args:
            data: Response from ScrapingDog API
            
        Returns:
            Dictionary with tweet data and media
        """
        try:
            tweet_data = {}
            
            tweet_result = data.get("data", {}).get("tweetResult", {}).get("result", {})
            legacy = tweet_result.get("legacy", {})
            core = tweet_result.get("core", {})
            
            tweet_data["text"] = legacy.get("full_text", "")
            
            tweet_data["id"] = legacy.get("id_str", "")

            extended_entities = legacy.get("extended_entities", {})

            if(extended_entities and "media" in extended_entities):
                media = extended_entities["media"][0]

                if media["type"] == "photo":
                    tweet_data["image"] = media.get("media_url_https", "")
                
            # if extended_entities and "media" in extended_entities:
            #     media = extended_entities["media"][0]
                
            #     if media["type"] == "video":
            #         video_info = media.get("video_info", {})
            #         video_variants = video_info.get("variants", [])
            #         video_url = ""
            #         max_bitrate = 0
            #         for variant in video_variants:
            #             if variant.get("content_type") == "video/mp4":
            #                 bitrate = variant.get("bitrate", 0)
            #                 if bitrate > max_bitrate:
            #                     max_bitrate = bitrate
            #                     video_url = variant[0]["url"]
                    
            #         if video_url:
            #             tweet_data["video"] = video_url,
            #     elif media["type"] == "photo":
            #         tweet_data["image"] = media.get("media_url_https", "")

            return tweet_data
                
        except Exception as e:
            logger.error(f"Error parsing ScrapingDog response: {str(e)}", exc_info=True)
            return {}