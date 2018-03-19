---
layout: page
title: "Toutes les recettes"
description: ""
---

<div class="related">
  <ul class="related-posts">
    {% for post in site.posts  %}
      <li>
        <h3>
          <a href="{{ post.url }}">
            {{ post.title }}
            <small>{{ post.date | date_to_string }}</small>
          </a>
        </h3>
      </li>
    {% endfor %}
  </ul>
</div>