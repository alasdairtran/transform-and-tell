import json
import logging
from datetime import datetime

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from tell.client import CaptioningClient

from .extractor import ExtractError, extract_article, get_urls

client = CaptioningClient(ip='localhost', port=5558, port_out=5559)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def visitor_ip_address(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')

    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


@csrf_exempt
def get_image_urls(request):
    query = json.loads(request.body)
    url = query['url'].strip()
    logger.info(f"{visitor_ip_address(request)} requests {url}")

    if not url:
        return JsonResponse({'error': 'The URL cannot be empty.'})
    if 'nytimes.com' not in url:
        return JsonResponse({'error': 'The URL must come from nytimes.com'})

    try:
        output = get_urls(url)
    except ExtractError as e:
        return JsonResponse({'error': str(e)})
    except Exception:
        return JsonResponse({'error': 'Cannot parse the article. Pick another URL.'})

    return JsonResponse(output)


@csrf_exempt
def post_caption(request):
    query = json.loads(request.body)

    article = extract_article(query['sections'], query['title'], query['pos'])
    output = client.parse([article])[0]
    output['caption'] = ''.join([a['tokens'] for a in output['attns'][0]])

    logger.info(f"Caption for {query['pos']}: {output['caption']}")

    data = {
        'title': article['title'],
        'image_url': article['image_url'],
        'generated_caption': output['caption'],
        'true_caption': article['true_caption'],
        'start': output['start'],
        'before': output['before'],
        'after': output['after'],
        'attns': output['attns'][0],
        'image': output['image'],
    }
    return JsonResponse(data)
