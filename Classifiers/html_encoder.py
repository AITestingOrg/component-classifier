import numpy as np
import pandas as pd
from html.parser import HTMLParser
        
class MyHTMLParser(HTMLParser):
    def clear(self):
        self.tags = {}
        
    def handle_starttag(self, tag, attrs):
        if tag in self.tags:
            self.tags[tag] = self.tags[tag] + 1
        else:
            self.tags[tag] = 1
            
    def getTags(self):
        return self.tags
        
class HTMLEncoder():
    def __init__(self):
         #each index corresponds to types of tags
        self.encodedlist = ['a','abbr','acronym','address','applet','area','article','aside',
        'audio','b','base','basefont','bdi','bdo','bgsound','big','blink','blockquote',
        'body','br','button','canvas','caption','center','cite','code','col','colgroup',
        'command','content','data','datalist','dd','del','details','dfn','dialog','dir',
        'div','dl','dt','element','em','embed','fieldset','figcaption','figure','font',
        'footer','form','frame','frameset','h1', 'h2', 'h3', 'h4', 'h5', 'h6','head',
        'header','hgroup','hr','html','i','iframe','image','img','input','ins','isindex',
        'kbd','keygen','label','legend','li','link','listing','main','map','mark',
        'marquee','menu','menuitem','meta','meter','multicol','nav','nobr','noembed',
        'noframes','noscript','object','ol','optgroup','option','output','p','param','path',
        'picture','plaintext','pre','progress','q','rp','rt','rtc','ruby','s','samp',
        'script','section','select','shadow','slot','small','source','spacer','span',
        'strike','strong','style','sub','summary','sup','svg','table','tbody','td',
        'template','textarea','tfoot','th','thead','time','title','tr','track','tt',
        'u','ul','var','video','wbr','xmp','nextid']
        
        self.types = np.zeros((len(self.encodedlist),), dtype=np.int)
        
    def encode(self, tag_dict):
        for tags in tag_dict:
            if tags not in self.encodedlist:
                continue
                
            index = self.encodedlist.index(tags)
            self.types[index] = tag_dict[tags]
        return self.types
    
    def clear(self):
        self.types = np.zeros((len(self.encodedlist),), dtype=np.int)
        
    def getencodedlist(self):
        return self.encodedlist    

    